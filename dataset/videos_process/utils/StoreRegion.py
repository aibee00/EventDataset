import os
import cv2
import sys
import yaml
import logging
import numpy as np
import glob
from tqdm import tqdm
from copy import deepcopy
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPolygon
from shapely.geometry.polygon import Polygon

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

class FastGridDistance(object):
    def __init__(self, coords, cell_size=10, max_distance=500):
        if isinstance(coords, Polygon):
            self._region = coords
            xs = coords.exterior.coords.xy[0].tolist()
            ys = coords.exterior.coords.xy[1].tolist()
            self._grid_info = self._cache_grid(
                [[x, y] for x, y in zip(xs, ys)], cell_size, max_distance)
        else:
            self._region = Polygon(coords)
            self._grid_info = self._cache_grid(coords, cell_size, max_distance)
        self._max_distance = max_distance

    @property
    def exterior(self):
        return self._region.exterior

    def _cache_grid(self, coords, cell_size, max_distance):
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]

        ret = {
            "x1": min(xs) - max_distance,
            "y1": min(ys) - max_distance,
            "x2": max(xs) + max_distance,
            "y2": max(ys) + max_distance,
            "cell_size": cell_size,
            "grid": None
        }

        h = np.ceil(float(ret['y2'] - ret['y1']) / cell_size).astype(int)
        w = np.ceil(float(ret['x2'] - ret['x1']) / cell_size).astype(int)
        grid = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            y = ret['y1'] + i * cell_size
            for j in range(w):
                x = ret['x1'] + j * cell_size
                pt = Point([x, y])
                d = self._region.exterior.distance(pt)
                if self._region.contains(pt):
                    grid[i, j] = -d
                else:
                    grid[i, j] = d
        ret['grid'] = grid
        return ret

    def distance(self, x):
        if x[0] < self._grid_info['x1'] or x[0] > self._grid_info['x2']:
            return self._max_distance
        if x[1] < self._grid_info['y1'] or x[1] > self._grid_info['y2']:
            return self._max_distance
        j = int(np.floor(float(x[0] - self._grid_info['x1']) / self._grid_info['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_info['y1']) / self._grid_info['cell_size']))
        return self._grid_info['grid'][i, j]

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
        grid = np.zeros((h, w), dtype=np.bool)
        for i in range(h):
            y = ret['y1'] + i * cell_size
            for j in range(w):
                x = ret['x1'] + j * cell_size
                if super(FastGridRegion, self).inside([x, y]):
                    grid[i, j] = True
        ret['grid'] = grid
        return ret

    def inside(self, x, t=None):
        if x[0] < self._grid_info['x1'] or x[0] > self._grid_info['x2']:
            return False
        if x[1] < self._grid_info['y1'] or x[1] > self._grid_info['y2']:
            return False
        j = int(np.floor(float(x[0] - self._grid_info['x1']) / self._grid_info['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_info['y1']) / self._grid_info['cell_size']))
        return self._grid_info['grid'][i, j]

class FastGridBand(object):
    def __init__(self, coords, margin=3.0, cell_size=0.2):
        self._region = Polygon(coords)
        self._margin = margin
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
        grid = np.zeros((h, w), dtype=np.bool)
        for i in range(h):
            y = ret['y1'] + i * cell_size
            for j in range(w):
                x = ret['x1'] + j * cell_size
                if self._inside([x, y]):
                    grid[i, j] = True
        ret['grid'] = grid
        return ret

    def _inside(self, x):
        '''
        Check if point x is inside of the region
        :param x:
        :return:
        '''
        pt = Point(x[0], x[1])
        d = self._region.exterior.distance(pt)
        return d < self._margin

    def distance(self, x):
        pt = Point(x[0], x[1])
        return self._region.exterior.distance(pt)

    def inside(self, x):
        if x[0] < self._grid_info['x1'] or x[0] > self._grid_info['x2']:
            return False
        if x[1] < self._grid_info['y1'] or x[1] > self._grid_info['y2']:
            return False
        j = int(np.floor(float(x[0] - self._grid_info['x1']) / self._grid_info['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_info['y1']) / self._grid_info['cell_size']))
        return self._grid_info['grid'][i, j]

def read_affine_file(filename):
    if not os.path.exists(filename):
        raise Exception("Affine transformation file does not exist")
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    H = fs.getNode('H').mat()
    invH = fs.getNode('invH').mat()
    return H, invH

def xy_to_floormap(H, pts):
    pts = np.expand_dims(pts, 2)
    xy_pts = np.squeeze(np.matmul(np.expand_dims(H, 0), pts), 2)
    return xy_pts

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

def project_img_to_floormap(extrinsic_file, intrinsic_file, pts, height=0, dist=30):
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

def draw_regions(floor_img_path, regions, color=(255, 255, 255)):
    floor_img = cv2.imread(floor_img_path)
    orig_img = np.copy(floor_img)
    for rgn in regions:
        x, y = rgn.exterior.coords.xy
        pts_floormap = np.array([d for d in zip(x, y)], dtype=np.int32)
        cv2.fillConvexPoly(floor_img, pts_floormap, color)
    floor_img = cv2.addWeighted(floor_img, 0.8, orig_img, 0.2, 0)
    return floor_img


class VisibleArea(object):
    def __init__(self, camera_info_path, to3d=False, door_cameras=[], max_distance_from_camera=15):
        self._margin = 50
        self._max_distance = max_distance_from_camera
        self._reference_height = 1.5

        self._camera_info_path = camera_info_path
        self._indoor_region = self._load_indoor_region(camera_info_path)
        self._camera_regions = self._camera_visibilities(camera_info_path)
        self._load_door_region(camera_info_path)

        # indoor intersection
        self._visible_regions = []
        for channel in self._camera_regions:
            region = self._camera_regions[channel].intersection(self._indoor_region)
            if region.area<=0:
                mlog.info("No indoor intersection for channel: {}".format(channel))
                continue
            if isinstance(region, Polygon):
                self._visible_regions.append(region)
            elif isinstance(region, MultiPolygon):
                self._visible_regions += list(region)

        # door facing FID intersection
        self._door_region = []
        for channel in self._coi+door_cameras:
            region = self._camera_regions[channel].intersection(self._indoor_region)
            if region.area<=0:
                mlog.info("No door intersection for channel of interest: {}".format(channel))
                continue
            if isinstance(region, Polygon):
                self._door_region.append(region)
            elif isinstance(region, MultiPolygon):
                self._door_region += list(region)

        if to3d:
            affine_file = os.path.join(camera_info_path, 'floorinfos', 'floor.yml')
            H, invH = read_affine_file(affine_file)
            for i, poly in enumerate(self._visible_regions):
                if not poly.exterior.coords:
                    continue
                x, y = poly.exterior.coords.xy
                pts_floormap = np.array([d for d in zip(x, y, [1 for _ in x])], dtype=np.int32)
                pts_3d = xy_to_floormap(invH, pts_floormap)
                self._visible_regions[i] = Polygon(pts_3d.tolist())
            for i, poly in enumerate(self._door_region):
                if not poly.exterior.coords:
                    continue
                x, y = poly.exterior.coords.xy
                # pts_floormap = np.array(zip(x, y, [1 for _ in x]), dtype=np.int32)
                pts_floormap = np.array([d for d in zip(x, y, [1 for _ in x])], dtype=np.int32)
                pts_3d = xy_to_floormap(invH, pts_floormap)
                self._door_region[i] = Polygon(pts_3d.tolist())
    @property
    def visible_regions(self):
        return self._visible_regions

    @property
    def door_regions(self):
        return self._door_region

    # @property
    # def entrance_cover_regions(self):
    #     return self._visible_regions

    def _camera_visibilities(self, camera_info_path):
        '''
        Get each camera's visible area using calibraiton
        :param camera_info_path:
        :return:
        '''
        affine_file = os.path.join(camera_info_path, 'floorinfos', 'floor.yml')
        H, _ = read_affine_file(affine_file)

        channels = sorted([
            os.path.basename(dname) for dname in glob.glob(os.path.join(camera_info_path, "regular/ch*"))
        ])

        mv_config = os.path.join(camera_info_path, "./mv/config.yaml")
        if not os.path.exists(mv_config):
            mv_config = os.path.join(camera_info_path, "./mv/store_v3_mv_config.yml")

        if os.path.exists(mv_config):
            # exclude DID channels and keep only fid
            with open(mv_config) as f:
                mv_cfg = yaml.load(f)

            channel_types = {}
            for d in mv_cfg["Camera"]:
                cname = d[0]
                ch, chtype = cname.split("-")[:2]
                channel_types[ch] = chtype
            channels = [ch for ch in channels if channel_types[ch] == "fid"]

        l_border = r_border = self._margin
        t_border = b_border = self._margin
        dist = self._max_distance
        height = self._reference_height

        ret = {}
        for channel in channels:
            intrinsic_file = os.path.join(camera_info_path, "regular", channel, 'intrinsic-calib.yml')
            extrinsic_file = os.path.join(camera_info_path, "regular", channel, 'extrinsic-calib.yml')

            intrinsic_fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
            image_width = intrinsic_fs.getNode('image_width').real()
            image_height = intrinsic_fs.getNode('image_height').real()

            border_pts = interpolate_points([np.array([[0 + l_border, 0 + t_border],
                                                       [0 + l_border, image_height - b_border],
                                                       [image_width - r_border, 0 + t_border],
                                                       [image_width - r_border, image_height - b_border]])])[0]
            pts_3d_world = project_img_to_floormap(
                extrinsic_file, intrinsic_file,
                border_pts, height=height, dist=dist)

            if pts_3d_world is None:
                continue

            pts_3d_world[:, 2] = 1
            ret[channel] = Polygon(xy_to_floormap(H, pts_3d_world)).convex_hull

        return ret

    def _load_door_region(self, camera_info_path):

        config_file = os.path.join(camera_info_path, "entrance_info.yaml")
        f = open(config_file)
        configs = yaml.load(f)
        f.close()
        assert 'door_region' in configs
        self._door = []
        for door in configs['door_region']:
            self._door.append(Polygon(door))
        assert 'door_region' in configs
        self._coi = configs['channels_of_interest']



    def _load_indoor_region(self, camera_info_path):
        '''
        Load indoor region data
        :param camera_info_path:
        :return:
        '''
        config_file = os.path.join(camera_info_path, "entrance_info.yaml")
        f = open(config_file)
        configs = yaml.load(f)
        f.close()
        assert 'indoor_region' in configs
        return Polygon(configs['indoor_region'])


VISIBLE_AREA = 'visible' # all FID visible area
ENTRANCE_COVER_AREA = 'entrance' # entrance path looking FIDs, defined by the dist from camera to door
DOOR_AREA = 'door' # door intersecting FIDs
class FastVisibleBoundary(object):
    def __init__(self, camera_info_path, margin=1.5, cell_size=0.2, roi=VISIBLE_AREA, max_distance_from_camera=25):
        '''
        For appearance based entrance/exit detection, it's important to have smaller roi. otherwise, there will be
        many ent_id generated
        :param camera_info_path:
        :param margin:
        :param cell_size:
        :param roi:
        '''
        self._visible_entrance = False # if this is False, many ent_id will be generated. Use this as an indicator
        va = VisibleArea(camera_info_path, True, max_distance_from_camera=max_distance_from_camera)
        if roi==VISIBLE_AREA:
            region = va.visible_regions
            area = sum([i.area for i in region])
            assert area>0, "no visible region for camera config: {}".format(camera_info_path)
        elif roi==DOOR_AREA:
            region = va.door_regions
            area = sum([i.area for i in region])
            if area>0:
                self._visible_entrance = True
        else:
            assert False, "not supported visible region"

        if not self._visible_entrance:
            # this could happen in the MRZOO case, set this as default to prevent bug
            region = va.visible_regions
            # assert False, "cannot proceed with empty visible roi for camera info: {} with roi option: {}".formt(
            #     camera_info_path, roi
            # )

        self._grid_inside, self._grid_boundary = self._fast_cache_grids(
            region,
            margin,
            cell_size
        )
        self._camera_info_path = camera_info_path

    @staticmethod
    def _fast_inside(x, regions, margin):
        '''
        Check whether a point x is near the boundary of visible area
        :param x:
        :param regions:
        :param margin:
        :return:
        '''
        for inside, bound in regions:
            isin = inside.inside(x)
            if isin:
                return True
        return False

    @staticmethod
    def _fast_near_boundary(x, regions, margin):
        '''
        Check whether a point x is near the boundary of visible area
        :param x:
        :param regions:
        :param margin:
        :return:
        '''
        count = 0
        any_near_boundary = False
        for inside, bound in regions:
            isin = inside.inside(x)
            isbound = bound.inside(x)
            if isin and not isbound:
                return False
            if isbound:
                any_near_boundary = True

        return any_near_boundary

    @staticmethod
    def _near_boundary(x, regions, margin):
        '''
        Check whether a point x is near the boundary of visible area
        :param x:
        :param regions:
        :param margin:
        :return:
        '''
        any_near_boundary = False
        for poly in regions:
            pt = Point(x[0], x[1])
            d = poly.exterior.distance(pt)
            if poly.contains(pt) and d > margin:
                # very well visible from one region
                return False
            if d < margin:
                any_near_boundary = True
        return any_near_boundary

    def _fast_cache_grids(self, regions, margin, cell_size):
        xs = []
        ys = []
        for poly in regions:
            if not poly.exterior.coords:
                continue
            x, y = poly.exterior.coords.xy
            xs += x
            ys += y

        grid_inside = {
            "x1": min(xs) - margin,
            "y1": min(ys) - margin,
            "x2": max(xs) + margin,
            "y2": max(ys) + margin,
            "cell_size": cell_size,
            "grid": None
        }

        grid_boundary = {
            "x1": min(xs) - margin,
            "y1": min(ys) - margin,
            "x2": max(xs) + margin,
            "y2": max(ys) + margin,
            "cell_size": cell_size,
            "grid": None
        }

        mlog.info("Create Fast region modules")
        temp = []
        for poly in tqdm(regions):
            if not poly.exterior.coords:
                continue
            x, y = poly.exterior.coords.xy
            temp.append([
                FastGridRegion([d for d in zip(x, y)], 0, cell_size),
                FastGridBand([d for d in zip(x, y)], margin, cell_size)
            ])
        regions = temp

        h = np.ceil(float(grid_inside['y2'] - grid_inside['y1']) / cell_size).astype(int)
        w = np.ceil(float(grid_inside['x2'] - grid_inside['x1']) / cell_size).astype(int)
        grid = np.zeros((h, w), dtype=np.bool)

        mlog.info("Caching inside grids")
        for i in tqdm(range(h)):
            y = grid_inside['y1'] + i * cell_size
            for j in range(w):
                x = grid_inside['x1'] + j * cell_size
                if self._fast_inside([x, y], regions, margin):
                    grid[i, j] = True
        grid_inside['grid'] = grid

        h = np.ceil(float(grid_boundary['y2'] - grid_boundary['y1']) / cell_size).astype(int)
        w = np.ceil(float(grid_boundary['x2'] - grid_boundary['x1']) / cell_size).astype(int)
        grid = np.zeros((h, w), dtype=np.bool)

        mlog.info("Caching boundary grids")
        for i in tqdm(range(h)):
            y = grid_boundary['y1'] + i * cell_size
            for j in range(w):
                x = grid_boundary['x1'] + j * cell_size
                if self._fast_near_boundary([x, y], regions, margin):
                    grid[i, j] = True
        grid_boundary['grid'] = grid
        return grid_inside, grid_boundary

    def _cache_grids(self, regions, margin, cell_size):
        xs = []
        ys = []
        for poly in regions:
            x, y = poly.exterior.coords.xy
            xs += x
            ys += y

        ret = {
            "x1": min(xs) - margin,
            "y1": min(ys) - margin,
            "x2": max(xs) + margin,
            "y2": max(ys) + margin,
            "cell_size": cell_size,
            "grid": None
        }

        h = np.ceil(float(ret['y2'] - ret['y1']) / cell_size).astype(int)
        w = np.ceil(float(ret['x2'] - ret['x1']) / cell_size).astype(int)
        grid = np.zeros((h, w), dtype=np.bool)
        for i in range(h):
            y = ret['y1'] + i * cell_size
            for j in range(w):
                x = ret['x1'] + j * cell_size
                if self._near_boundary([x, y], regions, margin):
                    grid[i, j] = True
        ret['grid'] = grid
        return ret

    def inside(self, x):
        if x[0] < self._grid_inside['x1'] or x[0] > self._grid_inside['x2']:
            return False
        if x[1] < self._grid_inside['y1'] or x[1] > self._grid_inside['y2']:
            return False
        j = int(np.floor(float(x[0] - self._grid_inside['x1']) / self._grid_inside['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_inside['y1']) / self._grid_inside['cell_size']))
        return self._grid_inside['grid'][i, j]

    def near_boundary(self, x):
        if x[0] < self._grid_boundary['x1'] or x[0] > self._grid_boundary['x2']:
            return False
        if x[1] < self._grid_boundary['y1'] or x[1] > self._grid_boundary['y2']:
            return False
        j = int(np.floor(float(x[0] - self._grid_boundary['x1']) / self._grid_boundary['cell_size']))
        i = int(np.floor(float(x[1] - self._grid_boundary['y1']) / self._grid_boundary['cell_size']))
        return self._grid_boundary['grid'][i, j]

    def get_affine_transform(self):
        '''
        Get transfomration data
        :return:
        '''
        camera_info_path = self._camera_info_path
        affine_file = os.path.join(camera_info_path, 'floorinfos', 'floor.yml')
        return read_affine_file(affine_file)

    def draw(self, show_bound=True):
        camera_info_path = self._camera_info_path
        floor_img_path = os.path.join(camera_info_path, 'floorinfos', 'floor.jpg')
        floor_img = cv2.imread(floor_img_path)
        height, width, _ = floor_img.shape
        sz = 4

        affine_file = os.path.join(camera_info_path, 'floorinfos', 'floor.yml')
        _, invH = read_affine_file(affine_file)
        cr = []
        for col in range(0, width, sz):
            for row in range(0, height, sz):
                cr.append([col, row, 1])
        pts_3d = xy_to_floormap(invH, np.array(cr))
        for pt, xy in zip(cr, pts_3d):
            if show_bound and self.near_boundary(xy):
                x1 = pt[0]
                x2 = min(pt[0] + sz, width)
                y1 = pt[1]
                y2 = min(pt[1] + sz, height)
                floor_img[y1:y2, x1:x2, 2] = 255 * 0.7 + floor_img[y1:y2, x1:x2, 2] * 0.3

            if self.inside(xy):
                x1 = pt[0]
                x2 = min(pt[0] + sz, width)
                y1 = pt[1]
                y2 = min(pt[1] + sz, height)
                floor_img[y1:y2, x1:x2, :] = 255 * 0.7 + floor_img[y1:y2, x1:x2, :] * 0.3
        return floor_img




if __name__ == "__main__":
    camera_info_path = "/root/CameraInfos/NWCL/guangzhou/yy/"
    camera_info_path = "/CameraInfos/NANLING/guangzhou/bydz/"
    # camera_info_path = "/root/CameraInfos/CTF/beijing/dfxtd/"
    # camera_info_path = "/root/CameraInfos/CTF/jinan/tf2f/"
    # camera_info_path = "/root/CameraInfos/CTF/dongguan/mygmc/"
    VISIBLE_AREA = 'visible'  # all FID visible area
    ENTRANCE_COVER_AREA = 'entrance'  # entrance path looking FIDs, defined by the dist from camera to door
    DOOR_AREA = 'door'  # door intersecting FIDs

    mod = FastVisibleBoundary(camera_info_path, cell_size=0.1, roi=VISIBLE_AREA)
    floor_img = mod.draw(True)
    cv2.imwrite("./tmp/boundary_bydz_visible.jpg", floor_img)

    # mod = FastVisibleBoundary(camera_info_path, cell_size=0.1, roi=ENTRANCE_COVER_AREA)
    # floor_img = mod.draw(True)
    # cv2.imwrite("/tmp/boundary_mygmc_entrance_cover.jpg", floor_img)

    mod = FastVisibleBoundary(camera_info_path, cell_size=0.1, roi=DOOR_AREA)
    floor_img = mod.draw(True)
    cv2.imwrite("./tmp/boundary_bydz_door.jpg", floor_img)