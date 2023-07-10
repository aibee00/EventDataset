import os
import sys
import cv2
import argparse
import numpy as np

from sys import platform

if platform == "darwin":
    import ruamel.yaml as ruamel_yaml
    import matplotlib

    matplotlib.use('TkAgg')
else:
    # import ruamel_yaml as ruamel_yaml
    import ruamel.yaml as ruamel_yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from camera import CameraInfoPacket

NUM_KPTS = 17
REF_PTS = 6

INDICES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)  # body
]

COLORS = [
    (0 / 255., 215 / 255., 255 / 255.), (0 / 255., 255 / 255., 204 / 255.),
    (0 / 255., 134 / 255., 255 / 255.), (0 / 255., 255 / 255., 50 / 255.),
    (77 / 255., 255 / 255., 222 / 255.), (77 / 255., 196 / 255., 255 / 255.),
    (77 / 255., 135 / 255., 255 / 255.), (191 / 255., 255 / 255., 77 / 255.),
    (77 / 255., 255 / 255., 77 / 255.), (0 / 255., 127 / 255., 255 / 255.),
    (255 / 255., 127 / 255., 77 / 255.), (0 / 255., 77 / 255., 255 / 255.),
    (255 / 255., 77 / 255., 36 / 255.)
]


def parse_arguments():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str,
                        default='/Users/zhanyu/Codes/data/3D/sample.image')
    parser.add_argument('--result_path', type=str,
                        default='/Users/zhanyu/Desktop/3D')
    parser.add_argument('--channel_pair', type=list,
                        default=[
                            ['ch01003_20190925094000', 'ch01005_20190925094000'],
                            ['ch01003_20190925094000', 'ch01006_20190925094000'],
                            ['ch01003_20190925094000', 'ch01007_20190925094000'],
                            ['ch01005_20190925094000', 'ch01006_20190925094000'],
                            ['ch01005_20190925094000', 'ch01007_20190925094000'],
                            ['ch01006_20190925094000', 'ch01007_20190925094000'],
                        ])
    parser.add_argument('--camera_info', type=str,
                        default='/Users/zhanyu/Codes/Python/CameraInfos')
    parser.add_argument('--store', type=str,
                        default='CTF/beijing/cytj')

    return parser.parse_args()


def load_camera_info(camera_info_dir):

    extrinsic_yaml_file = os.path.join(camera_info_dir, 'extrinsic-calib.yml')
    intrinsic_yaml_file = os.path.join(camera_info_dir, 'intrinsic-calib.yml')

    with open(extrinsic_yaml_file, 'r') as in_strm:
            extrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

            # R
            rotation_thetas = [float(item) for item in extrinsic_info['rvec']['data']]
            rotation_thetas_rows = int(extrinsic_info['rvec']['rows'])
            rotation_thetas_cols = int(extrinsic_info['rvec']['cols'])
            R = CameraInfoPacket.euler2rotation(
                np.array(rotation_thetas).reshape((rotation_thetas_rows, rotation_thetas_cols))
            )[0]

            # t
            translation_vects = [float(item) for item in extrinsic_info['tvec']['data']]
            translation_vects_rows = int(extrinsic_info['tvec']['rows'])
            translation_vects_cols = int(extrinsic_info['tvec']['cols'])
            t = np.array(translation_vects).reshape((translation_vects_rows, translation_vects_cols))

    with open(intrinsic_yaml_file, 'r') as in_strm:
        intrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

        # K
        K_rows = int(intrinsic_info['camera_matrix']['rows'])
        K_cols = int(intrinsic_info['camera_matrix']['cols'])
        K = np.array([float(item) for item in intrinsic_info['camera_matrix']['data']]).reshape((K_rows, K_cols))

        # distortion coefficient
        dist_coeff_rows = int(intrinsic_info['distortion_coefficients']['rows'])
        dist_coeff_cols = int(intrinsic_info['distortion_coefficients']['cols'])
        dist_coeff = np.array(
            [float(item) for item in intrinsic_info['distortion_coefficients']['data']]
        ).reshape((dist_coeff_rows, dist_coeff_cols))
    
    camera_info = CameraInfoPacket(P=None, K=K, R=R, t=t, dist_coeff=dist_coeff)
    return camera_info
    

def load_all_camera_info(camera_info_path, store):
    camera_info = dict()
    channels = os.listdir(os.path.join(camera_info_path, store, 'regular'))

    for channel in channels:
        extrinsic_yaml_file = os.path.join(camera_info_path, store, 'regular', channel, 'extrinsic-calib.yml')
        intrinsic_yaml_file = os.path.join(camera_info_path, store, 'regular', channel, 'intrinsic-calib.yml')

        with open(extrinsic_yaml_file, 'r') as in_strm:
            extrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

            # R
            rotation_thetas = [float(item) for item in extrinsic_info['rvec']['data']]
            rotation_thetas_rows = int(extrinsic_info['rvec']['rows'])
            rotation_thetas_cols = int(extrinsic_info['rvec']['cols'])
            R = CameraInfoPacket.euler2rotation(
                np.array(rotation_thetas).reshape((rotation_thetas_rows, rotation_thetas_cols))
            )[0]

            # t
            translation_vects = [float(item) for item in extrinsic_info['tvec']['data']]
            translation_vects_rows = int(extrinsic_info['tvec']['rows'])
            translation_vects_cols = int(extrinsic_info['tvec']['cols'])
            t = np.array(translation_vects).reshape((translation_vects_rows, translation_vects_cols))

        with open(intrinsic_yaml_file, 'r') as in_strm:
            intrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

            # K
            K_rows = int(intrinsic_info['camera_matrix']['rows'])
            K_cols = int(intrinsic_info['camera_matrix']['cols'])
            K = np.array([float(item) for item in intrinsic_info['camera_matrix']['data']]).reshape((K_rows, K_cols))

            # distortion coefficient
            dist_coeff_rows = int(intrinsic_info['distortion_coefficients']['rows'])
            dist_coeff_cols = int(intrinsic_info['distortion_coefficients']['cols'])
            dist_coeff = np.array(
                [float(item) for item in intrinsic_info['distortion_coefficients']['data']]
            ).reshape((dist_coeff_rows, dist_coeff_cols))

        camera_info[channel] = CameraInfoPacket(P=None, K=K, R=R, t=t, dist_coeff=dist_coeff)
    return camera_info


def create_camera_info(camera_info_path, store, channels):
    """

    :param camera_info_path:
    :param store:
    :param channels:
    :return:
    """
    camera_info = dict()

    for channel in channels:
        extrinsic_yaml_file = os.path.join(camera_info_path, store, 'regular', channel, 'extrinsic-calib.yml')
        intrinsic_yaml_file = os.path.join(camera_info_path, store, 'regular', channel, 'intrinsic-calib.yml')

        with open(extrinsic_yaml_file, 'r') as in_strm:
            extrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

            # R
            rotation_thetas = [float(item) for item in extrinsic_info['rvec']['data']]
            rotation_thetas_rows = int(extrinsic_info['rvec']['rows'])
            rotation_thetas_cols = int(extrinsic_info['rvec']['cols'])
            R = CameraInfoPacket.euler2rotation(
                np.array(rotation_thetas).reshape((rotation_thetas_rows, rotation_thetas_cols))
            )[0]

            # t
            translation_vects = [float(item) for item in extrinsic_info['tvec']['data']]
            translation_vects_rows = int(extrinsic_info['tvec']['rows'])
            translation_vects_cols = int(extrinsic_info['tvec']['cols'])
            t = np.array(translation_vects).reshape((translation_vects_rows, translation_vects_cols))

        with open(intrinsic_yaml_file, 'r') as in_strm:
            intrinsic_info = ruamel_yaml.load(in_strm, Loader=ruamel_yaml.RoundTripLoader)

            # K
            K_rows = int(intrinsic_info['camera_matrix']['rows'])
            K_cols = int(intrinsic_info['camera_matrix']['cols'])
            K = np.array([float(item) for item in intrinsic_info['camera_matrix']['data']]).reshape((K_rows, K_cols))

            # distortion coefficient
            dist_coeff_rows = int(intrinsic_info['distortion_coefficients']['rows'])
            dist_coeff_cols = int(intrinsic_info['distortion_coefficients']['cols'])
            dist_coeff = np.array(
                [float(item) for item in intrinsic_info['distortion_coefficients']['data']]
            ).reshape((dist_coeff_rows, dist_coeff_cols))

        camera_info[channel] = CameraInfoPacket(P=None, K=K, R=R, t=t, dist_coeff=dist_coeff)

    return camera_info


def distortPoint(kpts2d_reprj, K, dist_coeff):
    """

    :param kpts2d_reprj: numpy array with shape of (2, N)
    :param K: intrinsic matrix with shape of (3, 3)
    :param dist_coeff: distortion coefficient with shape of (1, 5)
    :return:
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    if dist_coeff.shape == (1, 5):
        k1 = dist_coeff[0, 0]
        k2 = dist_coeff[0, 1]
        p1 = dist_coeff[0, 2]
        p2 = dist_coeff[0, 3]
        k3 = dist_coeff[0, 4]
    else:
        k1 = dist_coeff[0]
        k2 = dist_coeff[1]
        p1 = dist_coeff[2]
        p2 = dist_coeff[3]
        k3 = dist_coeff[4]

    kpts2d_reprj_copy = kpts2d_reprj.copy()
    batch_size = kpts2d_reprj_copy.shape[1]
    for batch_idx in range(batch_size):
        # To relative coordinates
        x = (kpts2d_reprj_copy[0, batch_idx] - cx) / fx
        y = (kpts2d_reprj_copy[1, batch_idx] - cy) / fy

        r2 = x * x + y * y

        # Radial distortion
        xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

        # Tangential distortion
        xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
        yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

        # Back to absolute coordinates
        kpts2d_reprj[0, batch_idx] = xDistort * fx + cx
        kpts2d_reprj[1, batch_idx] = yDistort * fy + cy

    return kpts2d_reprj


def triangulation_and_reprojection(kpts2d_1, kpts2d_2, camera_1, camera_2, num_kpt=NUM_KPTS, should_distort=False):
    """

    :param kpts2d_1:
    :param kpts2d_2:
    :param camera_1:
    :param camera_2:
    :param num_kpt:
    :param should_distort:
    :return:
    """
    kpts2d_1_copy, kpts2d_2_copy = kpts2d_1.copy(), kpts2d_2.copy()

    if should_distort:
        assert kpts2d_1_copy.shape[:2] == kpts2d_2_copy.shape[:2]
        kpts_dim, kpts_batch = kpts2d_1_copy.shape[:2]
        kpts2d_1_copy = cv2.undistortPoints(kpts2d_1_copy, camera_1.K, camera_1.dist_coeff, dst=None,
                                            R=None, P=camera_1.K).reshape((kpts_batch, kpts_dim)).transpose(1, 0)
        kpts2d_2_copy = cv2.undistortPoints(kpts2d_2_copy, camera_2.K, camera_2.dist_coeff, dst=None,
                                            R=None, P=camera_2.K).reshape((kpts_batch, kpts_dim)).transpose(1, 0)

    # perform triangulation
    kpts4d_hom = cv2.triangulatePoints(camera_1.P, camera_2.P, kpts2d_1_copy, kpts2d_2_copy)  # 4 x 17
    kpts3d = (kpts4d_hom.T[:, :3] / kpts4d_hom.T[:, 3:]).T  # 3 x 17

    # perform re-projection
    kpts3d_hom = CameraInfoPacket.cart2hom(kpts3d.copy())
    kpts2d_1_reprj = camera_1.project(kpts3d_hom.copy())
    kpts2d_2_reprj = camera_2.project(kpts3d_hom.copy())

    if should_distort:
        kpts2d_1_reprj = distortPoint(kpts2d_1_reprj, camera_1.K, camera_1.dist_coeff)
        kpts2d_2_reprj = distortPoint(kpts2d_2_reprj, camera_2.K, camera_2.dist_coeff)

    reprj_error_view_1 = 0.
    reprj_error_view_all_1 = np.zeros(num_kpt)
    for ii in range(num_kpt):
        error = np.linalg.norm(kpts2d_1[:, ii] - kpts2d_1_reprj[:, ii])
        reprj_error_view_1 += error
        reprj_error_view_all_1[ii] = error
    reprj_error_view_1 /= num_kpt

    reprj_error_view_2 = 0.
    reprj_error_view_all_2 = np.zeros(num_kpt)

    for ii in range(num_kpt):
        error = np.linalg.norm(kpts2d_2[:, ii] - kpts2d_2_reprj[:, ii])
        reprj_error_view_2 += error
        reprj_error_view_all_2[ii] = error
    reprj_error_view_2 /= num_kpt

    return kpts3d, kpts2d_1_reprj, kpts2d_2_reprj, reprj_error_view_1, reprj_error_view_2, reprj_error_view_all_1, reprj_error_view_all_2


def reprojection(kpts3d, camera, should_distort=True):
    """

    :param kpts3d: shape(3,17), [[p1_x, p2_x, p3_x,...], [p1_y, p2_y, p3_y,...], [p1_z, p2_z, p3_z,...]]
    :param kpts2d_2:
    :param camera_1:
    :param camera_2:
    :param num_kpt:
    :param should_distort:
    :return:
    """

    # perform re-projection
    kpts3d_hom = CameraInfoPacket.cart2hom(kpts3d.copy())
    kpts2d_1_reprj = camera.project(kpts3d_hom.copy())

    if should_distort:
        kpts2d_1_reprj = distortPoint(kpts2d_1_reprj, camera.K, camera.dist_coeff)

    return kpts2d_1_reprj




# ========================================== New add by me ================================
def convert_2d_to_3d_v0(kpts2d):
    """
    :param kpts2d: shape(n, 2)
    :return: shape(3, n)
    """
    kpts3d = np.zeros((3, kpts2d.shape[0]))
    kpts3d[0, :] = kpts2d[:, 0]
    kpts3d[1, :] = kpts2d[:, 1]
    kpts3d[2, :] = 0

    return kpts3d


def convert_2d_to_3d_v1(kpts2d):
    """
    :param kpts2d: shape(n, 2)
    :return: shape(n, 3)
    """
    kpts3d = np.zeros((kpts2d.shape[0], 3))
    kpts3d[:, 0] = kpts2d[:, 0]
    kpts3d[:, 1] = kpts2d[:, 1]
    kpts3d[:, 2] = 1

    # kpts3d = np.append(kpts2d, np.ones((kpts2d.shape[0], 1)), axis=1)

    return kpts3d


def get_car_pose(car_pose):
    car_pose = json.loads(open(car_pose, 'r').read())
    car_coords = []
    for car in car_pose:
        car_coords.extend(car['cords'])
    return np.array(car_coords)


class Project3DTo2D():
    def __init__(self, camera_info_path, store) -> None:
        # Load camera info
        channels = os.listdir(os.path.join(camera_info_path, store, 'regular'))
        self._A = {}
        self._distCoeffs = {}
        self._rvec = {}
        self._tvec = {}
        self._fisheye = {}
        self._rmat = {}
        self._camera_center_world = {}

        floor_map_file = os.path.join(camera_info_path, store, 'floorinfos/floor_map.yml')
        floor_map = cv2.FileStorage(floor_map_file, cv2.FILE_STORAGE_READ)
        self.H_3to2map = floor_map.getNode('H_3to2map').mat()
        self.H_2mapto3 = floor_map.getNode('H_2mapto3').mat()

        self.ref_imgs = {}

        for channel in channels:
            extrinsic_file = os.path.join(camera_info_path, store, 'regular', channel, 'extrinsic-calib.yml')
            intrinsic_file = os.path.join(camera_info_path, store, 'regular', channel, 'intrinsic-calib.yml')

            extrinsic_fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
            intrinsic_fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)

            self._A[channel] = intrinsic_fs.getNode('camera_matrix').mat()
            self._distCoeffs[channel] = intrinsic_fs.getNode('distortion_coefficients').mat()
            self._rvec[channel] = extrinsic_fs.getNode('rvec').mat()
            self._tvec[channel] = extrinsic_fs.getNode('tvec').mat()
            self._fisheye[channel] = intrinsic_fs.getNode('is_fisheye').real()
            assert not self._fisheye[channel], "fish-eye cams are not supported yet"
            self._rmat[channel] = np.expand_dims(np.linalg.inv(cv2.Rodrigues(self._rvec[channel])[0]), 0)
            self._camera_center_world[channel] = np.matmul(self._rmat[channel], np.zeros((1, 3, 1)) - self._tvec[channel])

            # ref img
            ref_img_file = os.path.join(camera_info_path, store, 'regular', channel, 'ref.jpg')
            ref_img = cv2.imread(ref_img_file)
            self.ref_imgs[channel] = ref_img

    def project_floor_to_3d(self, pt2d):
        """ Project floor coordinates to 3d coordinates
        :param pt2d: shape(n, 2)
        :return: shape(n, 3)
        """
        if isinstance(pt2d, list):
            pt2d = np.array(pt2d)
        
        pt2d_hom = np.append(pt2d, np.ones((pt2d.shape[0], 1)), axis=1)
        # H_3to2map.shape(3, 3), pt2d_hom.shape(n, 3) -> pt3d.shape(n, 3)
        pt3d = np.array([np.dot(self.H_2mapto3, point_cam) for point_cam in pt2d_hom])
        return pt3d

    def project_3d_to_2d(self, pts3d, channel):
        """ project 3d point to 2d camera

        :pts3d: shape(n, 3)
        :channel: str, e.g. ch01001
        :return: shape(n, 2)
        """
        x = pts3d
        if len(pts3d.shape) == 1:
            x = pts3d[np.newaxis, :]
        # x = x[:, :, np.newaxis]
        return cv2.projectPoints(x, 
            self._rvec[channel], 
            self._tvec[channel], 
            self._A[channel], 
            self._distCoeffs[channel])[0]
    
    def plot_pt2d_on_ref_img(self, pt2d, channel):
        """ pt2d: 
            - 2d 平面的点坐标 或者 list(p1, p2)
        """
        print(f"Plotting pt: {pt2d}, shape: {pt2d}")
        if isinstance(pt2d, list) or (isinstance(pt2d, np.ndarray) and len(pt2d.shape) >= 2):
            pt2d = np.array(pt2d)

            # Plot on img
            for i in range(pt2d.shape[0]):
                try:
                    cv2.line(self.ref_imgs[channel], tuple(pt2d[i].astype(int)), tuple(pt2d[(i+1)%pt2d.shape[0]].astype(int)), (0, 0, 255), 2)
                    cv2.circle(self.ref_imgs[channel], tuple(pt2d[i].astype(int)), 20, (0, 0, 255), -1)
                except:
                    continue
        else:
            cv2.circle(self.ref_imgs[channel], tuple(pt2d.astype(int)), 20, (0, 0, 255), -1)
            # cv2.circle(self.ref_imgs[channel], (20, 30), 20, (0, 0, 255), -1)


def get_points(area_annotation):
    # Dummy assign
    kpts3d = np.array([[256, 788], [34, 998], [786, 998]])

    annos = []
    with open(area_annotation, 'r') as f:
        annos = json.load(f)
    
    front_door = annos['doors']['FRONT_DOOR:1']["coords"]
    # print(f"\t front_door: {front_door}")

    internal_door = annos['doors']['INTERNAL_DOOR:2']["coords"]
    print(f"\t internal_door: {internal_door}")

    # get car pose coords
    car_pose = "/ssd/wphu/work_dir/GACNE-guangzhou-xhthwk-20210717-v744.eventgpt/inputs/pose.json"
    car_coords = get_car_pose(car_pose)


    # kpts3d = convert_2d_to_3d_v1(np.concatenate((front_door, internal_door, car_coords), axis=0))
    kpts3d = np.concatenate((front_door, internal_door, car_coords), axis=0)
    print(f"\t kpts3d: {kpts3d.tolist()},\n\t shape: {kpts3d.shape} \n")

    return kpts3d


if __name__ == "__main__":
    # 设定np.print打印时不要用科学计数法格式
    np.set_printoptions(suppress=True)

    import json
    import yaml

    args = parse_arguments()

    print("test triangulation_and_reprojection")
    area_annotation_path = f"/ssd/wphu/StoreInfos/{args.store}/area_annotation.json"
    
    # test of reprojection
    camera_infos = load_all_camera_info(args.camera_info, args.store)
    print(camera_infos)

    project = Project3DTo2D(args.camera_info, args.store)
    
    for ch, camera in camera_infos.items():
        # kpts3d = np.array([[256, 788, 0], [34, 998, 0], [786, 998, 0]])
        kpts3d = get_points(area_annotation_path)

        kpts3d = project.project_floor_to_3d(kpts3d)

        kpts3d[:, 2] = 0
        print(f"point_3d_homogeneous: {kpts3d.tolist()}")

        if True:
            ######## Projection #######
            impts = project.project_3d_to_2d(kpts3d, ch)[:, 0, :]
            kpt1_2d = impts[:2]
            kpt2_2d = impts[2:4]
            kpt3_2d = impts[4:8]
            kpt4_2d = impts[8:12]

            # Clip
            img_height, img_width, _ = project.ref_imgs[ch].shape
            kpt1_2d = np.clip(kpt1_2d, 0, [img_width - 1, img_height - 1])
            kpt2_2d = np.clip(kpt2_2d, 0, [img_width - 1, img_height - 1])

            print(f"ch: {ch}, impts: {impts.tolist()}, \n kpt1_2d: {kpt1_2d}, \n kpt2_2d: {kpt2_2d}")

            project.plot_pt2d_on_ref_img(kpt1_2d, ch)
            project.plot_pt2d_on_ref_img(kpt2_2d, ch)
            project.plot_pt2d_on_ref_img(kpt3_2d, ch)
            project.plot_pt2d_on_ref_img(kpt4_2d, ch)

            # print(f"images WH: {project.ref_imgs[ch].shape}")

            # save img
            cv2.imwrite(f"test_reprojection_{ch}.jpg", project.ref_imgs[ch])
        else:
            def _convert_2d_to_3d(kpts2d):
                """
                :param kpts2d: shape(n, 3)
                :return: shape(3, n)
                """
                kpts3d = np.zeros((3, kpts2d.shape[0]))
                kpts3d[0, :] = kpts2d[:, 0]
                kpts3d[1, :] = kpts2d[:, 1]
                kpts3d[2, :] = kpts2d[:, 2]
                return kpts3d
            
            kpts3d = _convert_2d_to_3d(kpts3d)
            
            ######## Projection #######
            kpts2d_1_reprj = reprojection(kpts3d, camera_infos[ch], should_distort=True)
            if True or np.all(kpts2d_1_reprj > 0):
                print(f"ch: {ch}   kpts2d_1_reprj: {kpts2d_1_reprj.tolist()}, \n\t shape: {kpts2d_1_reprj.shape} \n\t {np.all(kpts2d_1_reprj > 0)}")
                # print(f"\t {np.all(kpts2d_1_reprj > 0)}")

                project.plot_pt2d_on_ref_img(kpts2d_1_reprj, ch)
                # save img
                cv2.imwrite(f"test_reprojection_{ch}.jpg", project.ref_imgs[ch])
