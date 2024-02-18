from copy import deepcopy
import os
import numpy as np
import yaml
from glob import glob
import json
import argparse
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import logging
import cv2
from random import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def interpolate_points(pts, num_interpolated_pts=100):
    out_pts = []
    for polygon in pts:
        x1 = min(polygon[:, 0])
        y1 = min(polygon[:, 1])
        x2 = max(polygon[:, 0])
        y2 = max(polygon[:, 1])
        grid_x = np.linspace(x1, x2, num_interpolated_pts)
        grid_y = np.linspace(y1, y2, num_interpolated_pts)
        polygon_pts = []
        pt_x, pt_y = np.meshgrid(grid_x, np.array([y1]))
        polygon_pts.append(np.reshape(np.dstack([pt_x, pt_y]), (-1, 2)))
        pt_x, pt_y = np.meshgrid(grid_x, np.array([y2]))
        polygon_pts.append(np.reshape(np.dstack([pt_x, pt_y]), (-1, 2)))
        pt_x, pt_y = np.meshgrid(np.array([x2]), grid_y)
        polygon_pts.append(np.reshape(np.dstack([pt_x, pt_y]), (-1, 2)))
        pt_x, pt_y = np.meshgrid(np.array([x1]), grid_y)
        polygon_pts.append(np.reshape(np.dstack([pt_x, pt_y]), (-1, 2)))
        polygon_pts = np.vstack(polygon_pts)
        out_pts.append(polygon_pts)
    return out_pts

def xy_to_floormap(H, pts):
    pts = np.expand_dims(pts, 2)
    xy_pts = np.squeeze(np.matmul(np.expand_dims(H, 0), pts), 2)
    return xy_pts

def camera_location(extrinsic_file):
    extrinsic_fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    rvec = extrinsic_fs.getNode('rvec').mat()
    tvec = extrinsic_fs.getNode('tvec').mat()
    assert (rvec.shape == (3, 1))
    assert (tvec.shape == (3, 1))
    pts_3d_world = np.matmul(np.expand_dims(np.linalg.inv(cv2.Rodrigues(rvec)[0]), 0), -tvec)
    return np.squeeze(pts_3d_world, 2)

def project_img_to_floormap(extrinsic_file, intrinsic_file, pts, height=0, dist=20):
    if not (os.path.exists(extrinsic_file) and os.path.exists(intrinsic_file)):
        raise Exception('Both extrinsic and intrinsic calibration required!')
    extrinsic_fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    intrinsic_fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
    A = intrinsic_fs.getNode('camera_matrix').mat()
    rvec = extrinsic_fs.getNode('rvec').mat()
    tvec = extrinsic_fs.getNode('tvec').mat()
    fisheye = intrinsic_fs.getNode('is_fisheye').real()
    assert (tvec.shape == (3, 1))
    assert (rvec.shape == (3, 1))

    if fisheye:
        distortion = intrinsic_fs.getNode('distortion_coefficients').mat()
        undist_pts = cv2.fisheye.undistortPoints(np.expand_dims(pts, 0), A, np.squeeze(distortion))
        undist_pts = np.squeeze(undist_pts, 0)
    else:
        undist_pts = cv2.undistortPoints(pts, A, np.zeros((5)))
    pts_3d_camera = cv2.convertPointsToHomogeneous(undist_pts).reshape(-1, 3, 1)
    pts_3d_world = np.matmul(np.expand_dims(np.linalg.inv(cv2.Rodrigues(rvec)[0]), 0), pts_3d_camera - tvec)
    camera_center_world = np.matmul(np.expand_dims(np.linalg.inv(cv2.Rodrigues(rvec)[0]), 0),
                                    np.zeros(pts_3d_camera.shape) - tvec)
    l = (height - camera_center_world[:, 2:3, :]) / (pts_3d_world - camera_center_world)[:, 2:3, :]
    norm = np.linalg.norm(pts_3d_world - camera_center_world, axis=1, keepdims=True)
    idx, _, _ = np.where(l * norm > dist)
    l[idx] = dist / norm[idx]
    l[l < 0] = 0
    pts_3d_world = camera_center_world + l * (pts_3d_world - camera_center_world)
    pts_3d_world = np.squeeze(pts_3d_world, 2)
    return pts_3d_world


def read_affine_file(filename):
    if not os.path.exists(filename):
        raise Exception("Affine transformation file does not exist")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    H = fs.getNode('H').mat()
    invH = fs.getNode('invH').mat()
    return H, invH


def plot_camera_coverage_regions(camerainfos, output_path, channels,
                                 image_width=2560, image_height=1440,
                                 l_border=50, r_border=50, t_border=50, b_border=50,
                                 dist=15):

    affine_file = os.path.join(camerainfos, 'floorinfos', 'floor.yml')
    H, _ = read_affine_file(affine_file)

    floor_img_path = os.path.join(camerainfos, 'floorinfos', 'floor.jpg')
    floor_img = cv2.imread(floor_img_path)

    channel_text_positions = {}
    for idx, ch in enumerate(channels):
        border_pts = interpolate_points([np.array([[0 + l_border, 0 + t_border],
                                                   [0 + l_border, image_height - b_border],
                                                   [image_width - r_border, 0 + t_border],
                                                   [image_width - r_border, image_height - b_border]])])[0]
        orig_img = np.copy(floor_img)
        # color = np.array(colormap(4 / len(channels))) * 255
        color = (255, 255, 255)

        pts_3d_world = project_img_to_floormap(
            os.path.join(ch, 'extrinsic-calib.yml'),
            os.path.join(ch, 'intrinsic-calib.yml'),
            border_pts, height=1.5, dist=dist)

        if pts_3d_world is None:
            continue
        pts_3d_world[:, 2] = 1
        pts_floormap = xy_to_floormap(H, pts_3d_world)

        channel_text_image_point = np.array([[0.0, image_height], [image_width, image_height]])
        channel_text_point_3d = project_img_to_floormap(
            os.path.join(ch, 'extrinsic-calib.yml'),
            os.path.join(ch, 'intrinsic-calib.yml'),
            channel_text_image_point, height=1.5, dist=dist)
        channel_text_point_3d[:, 2] = 1
        channel_text_floor_point = xy_to_floormap(
            H, channel_text_point_3d)
        text_pos = np.average(channel_text_floor_point, axis=0)
        channel_text_positions[ch] = text_pos

        pts_floormap = cv2.convexHull(pts_floormap.astype(np.float32))
        cv2.fillConvexPoly(floor_img, pts_floormap.astype(np.int32), color)

        floor_img = cv2.addWeighted(floor_img, 0.4, orig_img, 0.6, 0)
        
        cam_pos = camera_location(os.path.join(ch, 'extrinsic-calib.yml'))
        cam_pos[:, 2] = 1
        cam_pos = xy_to_floormap(H, cam_pos)[0]
        cv2.circle(floor_img, (int(cam_pos[0]), int(cam_pos[1])), 15, (0, 0, 255), -1)

    for ch in channel_text_positions:
        x_pos = int(channel_text_positions[ch][0])
        y_pos = int(channel_text_positions[ch][1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(floor_img, os.path.basename(ch), (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    # save the result
    store_tag = camerainfos.split('/')
    store_tag = '_'.join(store_tag[-3:])
    if len(channels) == 1:
        filepath = output_path + "/" + store_tag + '{}_{}m.png'.format(os.path.basename(channels[0]), dist)
    elif len(channels) == 2:
        filepath = output_path + "/" + store_tag + 'channel_pair_{}_{}_{}m.png'.format(
            os.path.basename(channels[0]), os.path.basename(channels[1]), dist)
    else:
        filepath = output_path + "/" + 'all_channels_{}m.png'.format(dist)
    mlog.info("save camera map in filepath {}".format(filepath))
    cv2.imwrite(filepath, floor_img)


class ShapelyRegion(object):
    def __init__(self, coords, margin=0):
        self._region = Polygon(coords)
        self._margin = margin

    @property
    def points(self):
        x, y = self._region.exterior.coords.xy
        return [(int(px), int(py)) for px, py in zip(x, y)]

    @property
    def centroid(self):
        return np.array([self._region.centroid.x, self._region.centroid.y], dtype=np.float32)

    def inside(self, x, t=None):
        '''
        Check if point x is inside of the region
        :param x:
        :return:
        '''
        pt = Point(x[0], x[1])
        if self._margin > 0:
            d = self._region.exterior.distance(pt)
            return d <= self._margin or self._region.contains(pt)
        else:
            return self._region.contains(pt)


class FastGridRegion(ShapelyRegion):
    def __init__(self, coords, margin=0, cell_size=50):
        super(FastGridRegion, self).__init__(coords, margin)
        self.coords = coords
        self.margin = margin
        self.cell_size = cell_size
        self._grid_info = self._cache_grid(coords, cell_size)

    def _cache_grid(self, coords, cell_size):
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]

        ret = {
            "x1": min(xs) - self._margin,
            "y1": min(ys) - self._margin,
            "x2": max(xs) + self._margin,
            "y2": max(ys) + self._margin,
            "cell_size": cell_size,
            "grid": None
        }

        h = np.ceil(float(ret['y2'] - ret['y1']) / cell_size).astype(int)
        w = np.ceil(float(ret['x2'] - ret['x1']) / cell_size).astype(int)
        grid = np.zeros((h, w), dtype=bool)
        for i in range(h):
            y = ret['y1'] + i * cell_size
            for j in range(w):
                x = ret['x1'] + j * cell_size
                if super(FastGridRegion, self).inside([x, y]):
                    grid[i, j] = True
        ret['grid'] = grid
        return ret

    def gen_grid_to_cameras_map(self, camera_info_path):
        self.engine = BestCameraFind(camera_info_path, max_dist=15)
        coords = self.coords
        cell_size = self.cell_size
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]

        ret = {
            "x1": min(xs) - self._margin,
            "y1": min(ys) - self._margin,
            "x2": max(xs) + self._margin,
            "y2": max(ys) + self._margin,
            "cell_size": cell_size,
            "grid": None
        }

        h = np.ceil(float(ret['y2'] - ret['y1']) / cell_size).astype(int)
        w = np.ceil(float(ret['x2'] - ret['x1']) / cell_size).astype(int)
        grid = {}
        max_x = 0.0
        max_y = 0.0
        for i in range(h):
            y = ret['y1'] + i * cell_size
            max_y = max(max_y, y)
            row = []
            for j in range(w):
                x = ret['x1'] + j * cell_size
                max_x = max(max_x, x)
                if super(FastGridRegion, self).inside([x, y]):
                    grid.setdefault(i,{})
                    grid[i][j] = self.get_camera_channels([x, y])
                if i in grid and j in grid[i] and grid[i][j]:
                    row.append(1)
                else:
                    row.append(0)
            # print("{:02}".format(i), row)
        ret['grid'] = grid
        return ret

    def get_camera_channels(self, loc):
        """ Get best cameras view by loc
        """
        best_cameras = self.engine(loc)
        return best_cameras

    def inside(self, x, t=None):
        if x[0] < self._grid_info['x1'] or x[0] > self._grid_info['x2']:
            return False
        if x[1] < self._grid_info['y1'] or x[1] > self._grid_info['y2']:
            return False
        j = int(np.floor(float(x[0] - self._grid_info['x1']) / self._grid_info['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_info['y1']) / self._grid_info['cell_size']))
        if i < 0 or i >= self._grid_info["grid"].shape[0]:
            return False
        if j < 0 or j >= self._grid_info["grid"].shape[1]:
            return False
        return self._grid_info['grid'][i, j]

    @property
    def grid_info(self):
        return self._grid_info


class BestCameraFind(object):
    """
    Given the camera info path and points in floor image map, find the matched camera that can get the best image of these points.
    """
    def __init__(self, store, max_dist=20):
        self.channel_infos = get_channels(store)
        affine_file = os.path.join(store, 'floorinfos', 'floor.yml')
        H, _ = read_affine_file(affine_file)
        l_border, r_border, t_border, b_border = 50, 50, 50, 50
        self._channles = {}
        for ch in self.channel_infos:
            channel, image_width, image_height = ch[0], ch[1], ch[2]
            border_pts = np.array([[0 + l_border, 0 + t_border],
                                                   [0 + l_border, image_height - b_border],
                                                   [image_width - r_border, image_height - b_border],
                                                   [image_width - r_border, 0 + t_border]])
            pts_3d_world = project_img_to_floormap(
                os.path.join(channel, 'extrinsic-calib.yml'),
                os.path.join(channel, 'intrinsic-calib.yml'),
                border_pts, height=1.5, dist=max_dist)
            
            ch_name = channel.split("/")[-1]
            if pts_3d_world is None:
                mlog.warning("Channel {} donot have any FOV".format(ch_name))
                continue
            pts_3d_world[:, 2] = 1
            pts_floormap = xy_to_floormap(H, pts_3d_world)
            cam_pos = camera_location(os.path.join(channel, 'extrinsic-calib.yml'))
            cam_pos[:, 2] = 1
            cam_pos = xy_to_floormap(H, cam_pos)[0]
            self._channles.setdefault(ch_name, {})
            self._channles[ch_name]["region"] = Polygon(pts_floormap)
            self._channles[ch_name]["cam_pos"] = [cam_pos[0], cam_pos[1]]

    def __call__(self, point):
        """
        points = {timestamp: [x, y]}, x,y is the coordinates in floormap.
        """
        best_cameras = self._find_best_view(point)
        return best_cameras

    def _find_best_view(self, loc):
        best_cameras = {}
        pt = Point(loc[0], loc[1])
        for ch_name in self._channles:
            if self._channles[ch_name]["region"].contains(pt):
                distance_to_camera = np.linalg.norm(np.array(loc) - np.array(self._channles[ch_name]["cam_pos"]))
                best_cameras[distance_to_camera] = ch_name
        best_cameras = sorted(best_cameras.items(), key=lambda x : x[0])
        best_cameras = [i[1] for i in best_cameras]
        return best_cameras

def get_channels(root_path):
    def opencv_matrix(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

    channel_folders = glob(os.path.join(root_path, 'regular', '*')) + glob(
        os.path.join(root_path, 'fisheye', '*', 'pano'))

    channel_types = {}
    mv_config = os.path.join(root_path, "./mv/config_nanling_vw_poc_reid.yaml")
    if not os.path.exists(mv_config):
        mv_config = os.path.join(root_path, "./mv/store_v3_mv_config.yml")
    if os.path.exists(mv_config):
        # exclude 10.7.152.103 channels and keep only fid
        with open(mv_config) as f:
            mv_cfg = yaml.load(f, Loader=yaml.FullLoader)

        for d in mv_cfg["Camera"]:
            cname = d[0]
            ch, chtype = cname.split("-")[:2]
            channel_types[ch] = chtype

    channels = []
    for ch_dir in channel_folders:
        if channel_types:
            cname = ch_dir.split("/")[-1]
            if cname not in channel_types or channel_types[cname] != "fid":
                continue

        files = os.listdir(ch_dir)
        if 'extrinsic-calib.yml' not in files or 'intrinsic-calib.yml' not in files:
            continue
        intrinsic_calibration_path = os.path.join(ch_dir, 'intrinsic-calib.yml')
        with open(intrinsic_calibration_path) as intrinsic_file:
            skip_lines = 2
            for i in range(skip_lines):
                _ = intrinsic_file.readline()
            intrinsic_params = yaml.load(intrinsic_file, Loader=yaml.FullLoader)
            image_width, image_height = intrinsic_params['image_width'], intrinsic_params['image_height']
            channels.append([ch_dir, image_width, image_height])
    return channels


def get_best_camera_views(x, grid_cameras_map):
    if isinstance(list(grid_cameras_map['grid'].keys())[0], str):
        grid_info = deepcopy(grid_cameras_map)
        grid_info['grid'] = {}
        for r in grid_cameras_map['grid'].keys():
            grid_info['grid'].setdefault(int(r), {})
            for c in grid_cameras_map['grid'][r].keys():
                grid_info['grid'][int(r)][int(c)] = grid_cameras_map['grid'][r][c]
    else:
        grid_info = grid_cameras_map
    if x[0] < grid_info['x1'] or x[0] > grid_info['x2']:
        return []
    if x[1] < grid_info['y1'] or x[1] > grid_info['y2']:
        return []
    j = int(np.floor(float(x[0] - grid_info['x1']) / grid_info['cell_size']))
    i = int(np.floor(float(x[1] - grid_info['y1']) / grid_info['cell_size']))
    if i < 0 or i not in grid_info["grid"].keys():
        return []
    if j < 0 or j not in grid_info["grid"][i].keys():
        return []
    return grid_info['grid'][i][j]


def plot_point_on_floor(args, point, best_cameras):
    """ For debug if you want to check if the auto-selected camera-views is right
    The input point is the point on floor map that you want to check.
    Input: 
        - point: duple, (x,y)
    """
    floormap = os.path.join(args.camera_info_path, 'floorinfos', 'floor.jpg')
    img = cv2.imread(floormap)
    cv2.circle(img, point, 30, (255,0,0), -1)
    
    max_dist = 15
    affine_file = os.path.join(args.camera_info_path, 'floorinfos', 'floor.yml')
    H, _ = read_affine_file(affine_file)
    engine = BestCameraFind(args.camera_info_path, max_dist=max_dist)
    l_border, r_border, t_border, b_border = 50, 50, 50, 50
    for ch in engine.channel_infos:
        channel, image_width, image_height = ch[0], ch[1], ch[2]
        border_pts_view = interpolate_points([np.array([[0 + l_border, 0 + t_border],
                                                [image_width - r_border, 0 + t_border],
                                                [image_width - r_border, 0 + t_border],
                                                [0 + l_border, image_height - b_border]])])[0]
        pts_3d_world_view = project_img_to_floormap(
            os.path.join(channel, 'extrinsic-calib.yml'),
            os.path.join(channel, 'intrinsic-calib.yml'),
            border_pts_view, height=1.5, dist=max_dist)
        cam_pos = camera_location(os.path.join(channel, 'extrinsic-calib.yml'))
        cam_pos[:, 2] = 1
        cam_pos = xy_to_floormap(H, cam_pos)[0]
        pts_3d_world_view[:, 2] = 1
        pts_floormap_view = xy_to_floormap(H, pts_3d_world_view)
        area_annotation_dir = os.path.join(args.store_infos_path, 'area_annotation.json')
        area_annotation = json.loads(open(area_annotation_dir, 'rb').read())
        coords = area_annotation["region_areas"]["STORE:0"]["coords"]
        color = (0 * random(), 255 * random(),255 * random())
        ch_name = channel.split("/")[-1]
        if ch_name in best_cameras:
            cv2.circle(img, (int(cam_pos[0]), int(cam_pos[1])), 20, (0,0,255), -1)
            cv2.putText(img, ch_name, (int(cam_pos[0]), int(cam_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            for point in pts_floormap_view:
                x,y = int(point[0]), int(point[1])
                cv2.circle(img, (x,y), 8, color, -1)
        for point in coords:
            x,y = int(point[0]), int(point[1])
            cv2.circle(img, (x,y), 10, (0,0,255), 4)
        # cv2.circle(img, (coords[0][0] + 41 * 50, coords[0][1] + 11 * 50), 50, (0,0,255), 8)
        # cv2.circle(img, (2539, 604), 50, (255,0,0), 8)
        # cv2.circle(img, (coords[0][0]+0 * 50, coords[0][1]+0 * 50), 50, (0,50,65), 8)
        # cv2.circle(img, (coords[0][0]+51 * 50, coords[0][1]+39 * 50), 50, (0,150,65), 8)
    file_name = os.path.join(args.save_path, "floor.jpg")
    cv2.imwrite(file_name, img)


def gen_grid_cameras_map(camera_info_path, store_infos_path):
    area_annotation_dir = os.path.join(store_infos_path, 'area_annotation.json')
    area_annotation = json.loads(open(area_annotation_dir, 'rb').read())

    # Gen grid to cameras map
    coords = area_annotation["region_areas"]["STORE:0"]["coords"]
    store_region = FastGridRegion(coords)
    grid_cameras_map = store_region.gen_grid_to_cameras_map(camera_info_path)
    return grid_cameras_map


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_info_path', type=str, help="camera_info_path", default="/ssd/wphu/CameraInfos/GACNE/guangzhou/xhthwk")
    parser.add_argument('--store_infos_path', type=str, help="store_infos_path", default="/ssd/wphu/StoreInfos/GACNE/guangzhou/xhthwk")
    parser.add_argument('--save_path', type=str, help="output path", default="./")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_dir = os.path.join(args.save_path, './grid_cameras_map.json')
    grid_cameras_map = gen_grid_cameras_map(args.camera_info_path, args.store_infos_path)
    with open(save_dir, 'w') as f:
        json.dump(grid_cameras_map, f, indent=2)

    """ Take an example for show how to use
    """
    grid_cameras_map = json.loads(open(save_dir, 'rb').read())
    test_point = (1800,1600)
    best_cameras = get_best_camera_views(test_point, grid_cameras_map)
    print(best_cameras)

    # debug: viz channel polygon
    plot_point_on_floor(args, test_point, best_cameras)


