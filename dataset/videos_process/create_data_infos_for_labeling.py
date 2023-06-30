import os
import os.path as osp
import json
import numpy as np
import bisect
import argparse
from gen_grid_cameras_map import FastGridRegion, get_best_camera_views
from glob import glob
import multiprocessing
import cv2
import shutil
from tqdm import tqdm
from utils.Store import Store, FastGrid, check_car_pose_poly_order
from dataset.SharedUtils import TrackWrapper, replace_mask, replace_rule, run_system_command, get_pose_json
from aibee_hdfs import hdfscli


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_info_path', type=str, help="camera_info_path", default="/ssd/wphu/CameraInfos/")
    parser.add_argument('--store_infos_path', type=str, help="output path", default="/ssd/wphu/StoreInfos/")
    parser.add_argument('--save_path', type=str, help="output path", default="./data_infos_labeling/")
    parser.add_argument('--video_path', type=str, help="output path", default="/ssd/wphu/videos/")
    parser.add_argument('--n_process', type=int, help="num process for loading pid", default=4)
    parser.add_argument('--data_path', type=str, help='path for features of all bmks', default='/ssd/wphu/release.v7.5.0/Dataset/data')
    parser.add_argument('--output_path', type=str, help='output pb path', required=False, default="/ssd/wphu/release.v7.5.0/output")
    parser.add_argument('--diplacement_consistency_window', type=int, help='diplacement_consistency_window', default=5)
    parser.add_argument("--work_dir", type=str, help="path of work_dir for store_pipeline", default='/ssd/wphu/work_dir/')
    parser.add_argument("--tag", type=str, help="tag of run version", default='v7.5.0.0408')
    parser.add_argument('--username', type=str, required=False, default="wphu", help='job owner name')
    return parser.parse_args()

args = parse_arguments()
assert args.username is not None, \
    "When running in non local mode, username is required"
keytab = "/home/{}/{}.keytab".format(args.username, args.username)
hdfscli.initKerberos(keytab, args.username)
client = hdfscli.HdfsClient(user=args.username)

def get_loc(pid_file):
    # pid_file = os.path.join(xy, pid + ".pb")
    if not os.path.exists(pid_file):
        print(f"{pid_file} not found!")
        return {}
    tkr = TrackWrapper.from_pb_file(pid_file)
    pid2xy = {'loc': {}, 'vel': {}, 'acc': {}}
    for i in range(len(tkr.ts_vec)):
        pid2xy['loc'][int(tkr.ts_vec[i])] = tkr.pts_vec[i]
        pid2xy['vel'][int(tkr.ts_vec[i])] = tkr.vel_vec[i]
        pid2xy['acc'][int(tkr.ts_vec[i])] = tkr.acc_vec[i]
    return pid2xy


def ts_to_string(ts, sec_size=1, sep=":"):
    # from 40ms tic to XXhYYmZZs
    h = int(float(ts) / (sec_size * 60 * 60))
    m = int(float(ts) / (sec_size * 60)) % 60
    s = int(float(ts) / sec_size) % 60
    return "{0:02d}{3}{1:02d}{3}{2:02d}".format(h, m, s, sep)

def hmsstr_to_ts(str):
    h = str[0:2]
    m = str[2:4]
    s = str[4:6]
    return int(h)*3600 + int(m)*60 + int(s)

class DataGen(object):
    def __init__(self, args, debug_viz_infos_path):
        self.args = args
        self.debug_viz_infos_path = debug_viz_infos_path
        self.store = None
        self.debug_viz_infos = None
        self.meter2pixel = None
        self.grid_cameras_map = None

    def _update_bmk(self, bmk):
        """ Update store by bmk
        """
        if not bmk:
            return
        args = self.args
        date = bmk.split('-')[-1]

        print("Reading per PID tracks")
        xy_path = osp.join(args.work_dir, "{}-{}".format(bmk, args.tag), "output_data")
        xy_dir = osp.join(xy_path, "new_xy")
        if not os.path.exists(xy_dir):
            run_system_command("mkdir -p {}".format(xy_dir))
            run_system_command("tar xvf {} -C {}".format(osp.join(xy_path, "new_xy.tar.gz"), xy_dir))
        pid_files = sorted(glob(osp.join(xy_dir, '*.pb')))
        pool = multiprocessing.Pool(args.n_process)
        locs = pool.map(get_loc, pid_files)
        self.pid_locs_raw = {}  # get the origin loc infos for interactions
        for pid_file, loc in zip(pid_files, locs):
            pid = osp.splitext(osp.basename(pid_file))[0]
            self.pid_locs_raw[pid] = loc
        
        # Update store debug_viz_infos for current bmk
        print("Loading meter pixel conversion")
        bmk_subpath = '/'.join(bmk.split('-')[:3])
        storeInfo = os.path.join(self.args.store_infos_path, bmk_subpath)
        cameraInfo = os.path.join(self.args.camera_info_path, bmk_subpath)
        print("storeInfo:", storeInfo, bmk, bmk_subpath)

        # Create local output directory
        store_tag = '_'.join(bmk.split('-')[:3])
        output_dir = '/tmp/{}/{}/'.format(store_tag, date)
        if osp.isdir(output_dir):
            run_system_command('rm -r {}'.format(output_dir))
        run_system_command('mkdir -p {}'.format(output_dir))

        store_rule_fpath = osp.join(storeInfo, 'event_rules.json')
        new_rule_fpath = osp.join(output_dir, 'event_rules.json')
        shutil.copy(store_rule_fpath, new_rule_fpath)
        store_rule_fpath = new_rule_fpath

        store_anno_fpath = osp.join(storeInfo, 'area_annotation.json')
        new_anno_fpath = osp.join(output_dir, 'area_annotation.json')
        shutil.copy(store_anno_fpath, new_anno_fpath)
        store_anno_fpath = new_anno_fpath

        # Check if daily mask input is available, and update counter masks
        daily_mask_path = os.path.join(args.work_dir, "{}-{}".format(bmk, args.tag), "inputs/pose.json")
        if daily_mask_path:
            args.daily_mask_path = daily_mask_path
        if args.daily_mask_path and args.daily_mask_path.lower() != 'none':
            print("Using daily mask at %s", args.daily_mask_path)
            local_daily_mask_path = \
                osp.join(self.args.output_path, osp.basename(args.daily_mask_path))
            run_system_command(f"cp {args.daily_mask_path} {local_daily_mask_path}")

            # check the correctness of polygon orders
            if not check_car_pose_poly_order(local_daily_mask_path):
                car_pose = get_pose_json(store_tag, date)
                with open(local_daily_mask_path, "wt") as f:
                    f.write(json.dumps(car_pose, sort_keys=True, indent=2))
                if not car_pose or not check_car_pose_poly_order(local_daily_mask_path):
                    assert False, "Cannot proceed! Car pose with wrong order: Door estimation requires points in clock-wise order"
            # Region type defaults to CAR: 2 (last argument)
            replace_mask(store_anno_fpath, local_daily_mask_path, True, 2)
            replace_rule(store_rule_fpath, local_daily_mask_path, True)

        fs = cv2.FileStorage(osp.join(cameraInfo, 'floorinfos', 'floor.yml'), cv2.FILE_STORAGE_READ)
        self.meter2pixel = fs.getNode("Scale").real()
        self.store = Store(output_dir, meter2pixel=self.meter2pixel)
        # save out counter id and shape poly
        counter_save = {}
        for reg_type, reg_dict in self.store.regions.items():
            for c in reg_dict:
                counter_save[reg_type + ':' + c] = {'x': [], 'y': []}
                if isinstance(reg_dict[c], FastGrid):
                    x, y = reg_dict[c].region.exterior.coords.xy
                else:
                    x, y = reg_dict[c].exterior.coords.xy
                counter_save[reg_type + ':' + c]['x'] = x.tolist()
                counter_save[reg_type + ':' + c]['y'] = y.tolist()
        self.debug_viz_infos = self._get_debug_viz_infos(bmk)

        # Gen grid to cameras map
        bmk_subpath = '/'.join(bmk.split('-')[:3])
        area_annotation_dir = os.path.join(self.args.store_infos_path, bmk_subpath, 'area_annotation.json')
        area_annotation = json.loads(open(area_annotation_dir, 'rb').read())
        coords = area_annotation["region_areas"]["STORE:0"]["coords"]
        store_region = FastGridRegion(coords)
        self.grid_cameras_map = store_region.gen_grid_to_cameras_map(os.path.join(self.args.camera_info_path, bmk_subpath))


    def get_data_infos(self, bmk):
        """ This version's definition:
        train/test data come from different benchmarks
        For example: 
            - train data : HONGQI-beijing-fkwd-2022019, VOLVO-jinan-xfh-20210617
            - test data  : GACNE-guangzhou-xhthwk-20210717
        V3 New Add:
            - Add diplacement_consistency_window [-5s,5s]
            - Loading data from data_dict.json
        Return:
            dict[pid][staff][ts]: {"consistent_window_ts": ts_list, "feature": feat_vec, "video_path": str}
        """
        data_train = {}
        self._update_bmk(bmk)

        file_dir = osp.join(self.args.data_path, bmk, 'data_dict.json')  # use strict condition to gen tpfp
        if osp.exists(file_dir):
            print("Loading features from {}".format(file_dir))
            data = json.loads(open(file_dir, 'r').read())
            feature_pos = self.parse_consistency_window_from_data(data["feature_pos"], bmk)
            feature_neg = self.parse_consistency_window_from_data(data["feature_neg"], bmk)
            data_train.update(feature_pos)
            for pid in tqdm(list(feature_neg.keys())):
                for staff in feature_neg[pid]:
                    data_train.setdefault(pid, {})
                    data_train[pid].setdefault(staff, {})
                    data_train[pid][staff].update(feature_neg[pid][staff])
        else:
            print("{} not found!".format(file_dir))
        return data_train

    def parse_consistency_window_from_data(self, data, bmk):
        """ Parse consistent window from data dict
        """
        offset = self.args.diplacement_consistency_window
        features = {}
        num = 0
        for pid in tqdm(list(data.keys())):
            for staff in data[pid]:
                common_ts = sorted([int(t) for t in data[pid][staff].keys()])
                for i, ts in enumerate(common_ts):
                    ts_left = []
                    ts_right = []
                    i1 = bisect.bisect_left(common_ts, ts - offset)
                    if i1 < i:
                        ts_left = [common_ts[i] for i in range(i1, i)]  # 5s
                    else:
                        ts_left = [ts]
                    i2 = bisect.bisect_left(common_ts, ts + offset)
                    if i < i2 - 1:
                        ts_right = [common_ts[i] for i in range(i, i2)]  # 5s
                    else:
                        ts_right = [ts]
                    feat_vec = []
                    ts_list = set(ts_left + ts_right)

                    # if len(ts_list) < 2 * offset:  # Filter out ts num < 10s
                    #     continue
                    
                    if False:  # Using statistic feature
                        for t in ts_list:
                            feat_vec.append(data[pid][staff][str(t)][:-1])
                        if len(ts_list) < 2 * offset:
                            feat_vec_interp = self._interpolation(pid, staff, data, ts, ts_list, offset)
                            feat_vec = []
                            for f_vec in feat_vec_interp:
                                feat_vec.append(f_vec)
                        # Flatten after standard
                        # feat_vec = preprocessing.StandardScaler().fit_transform(feat_vec).reshape(-1).tolist()
                        # feat_vec = preprocessing.Normalizer().fit_transform(feat_vec).reshape(-1).tolist()
                        # feat_vec = preprocessing.Normalizer().fit_transform(feat_vec).tolist()
                        feat_vec = self._get_statistic_feature(feat_vec)
                        
                    else:
                        for t in ts_list:
                            feat_vec.extend(data[pid][staff][str(t)][:-1])
                        if len(ts_list) < 2 * offset:
                            feat_vec_interp = self._interpolation(pid, staff, data, ts, ts_list, offset)
                            feat_vec = []
                            for f_vec in feat_vec_interp:
                                feat_vec.extend(f_vec)

                        use_both_raw_and_statistic_feat = False
                        if use_both_raw_and_statistic_feat:
                            feat_vec_stat = []
                            for t in ts_list:
                                feat_vec_stat.append(data[pid][staff][str(t)][:-1])
                            if len(ts_list) < 2 * offset:
                                feat_vec_interp = self._interpolation(pid, staff, data, ts, ts_list, offset)
                                feat_vec_stat = []
                                for f_vec in feat_vec_interp:
                                    feat_vec_stat.append(f_vec)
                            feat_vec_stat = self._get_statistic_feature(feat_vec_stat)
                            feat_vec.extend(feat_vec_stat)

                        use_combined_feature = True
                        if use_combined_feature:
                            feat_vec_stat = []
                            for t in ts_list:
                                feat_vec_stat.append(data[pid][staff][str(t)][:-1])
                            if len(ts_list) < 2 * offset:
                                feat_vec_interp = self._interpolation(pid, staff, data, ts, ts_list, offset)
                                feat_vec_stat = []
                                for f_vec in feat_vec_interp:
                                    feat_vec_stat.append(f_vec)
                            feat_vec.extend(self._get_combine_feature(feat_vec_stat))

                    feat_vec.append(data[pid][staff][str(ts)][-1])  # append label at ts
                    features.setdefault(pid, {})
                    features[pid].setdefault(staff, {})
                    features[pid][staff].setdefault(ts, {})
                    features[pid][staff][ts]["consistent_window_ts"] = list(ts_list)
                    features[pid][staff][ts]["feature"] = feat_vec
                    features[pid][staff][ts]["video_path"] = self._get_video_path(bmk, pid, staff, ts, ts_list)
                    num += 1
        print("Get {} samples!".format(num))
        return features

    @staticmethod
    def _get_statistic_feature(feat_vec):
        """ Statistic feature: min_val, max_val, median_val, norm_val
        """
        feat_arr = np.array(feat_vec)
        features = np.vstack((np.min(feat_arr, axis=0), np.max(feat_arr, axis=0),
                            np.median(feat_arr, axis=0), np.linalg.norm(feat_arr, axis=0)))
        features = features.T.reshape(-1)
        return features.tolist()

    @staticmethod
    def _get_combine_feature(feat_vec):
        """ Statistic feature: min_val, max_val, median_val, norm_val
        """
        feat_new = []
        feat_arr = np.array(feat_vec)
        all_feats = ["dist", "common_region_w", "move_bonus", "customer_incar", "staff_incar", "discount", "md",
                    "a_look_b", "b_look_a", "cofocal", "view_align", "distance_3d"]
        dist = np.mean(sorted(feat_arr[:, 0])[:3])
        dist_3d = np.mean(sorted(feat_arr[:, -1])[:3])
        feat_new.append(dist)
        feat_new.append(dist_3d)
        max_feat_set = ["common_region_w", "move_bonus", "customer_incar", "staff_incar", "discount", "md", "a_look_b", "b_look_a", "cofocal", "view_align"]
        for feat in max_feat_set:
            idx = all_feats.index(feat)
            feat_val = feat_arr[:, idx]
            mean_max = np.mean(sorted(feat_val, reverse=True)[:3])
            feat_new.append(mean_max)
        # features = np.vstack((np.median(feat_arr, axis=0), np.linalg.norm(feat_arr, axis=0))).T.reshape(-1).tolist()
        # return feat_new + features
        return feat_new

    def _interpolation(self, pid, staff, data, cur_ts, ts_list, diplacement_consistency_window):
        """
        For make sure all feature have same length, we need interpolation on short feature
        """
        ts_list = sorted(ts_list)
        # print("cur_ts: {}, ts_list: {}".format(cur_ts, ts_list))
        feature_n = []
        # step 1, Process for interpolation from center
        feature_n.append(data[pid][staff][str(ts_list[0])][:-1])
        for ts in ts_list[1:]:
            i = ts_list.index(ts)
            if ts - ts_list[i-1] > 1:
                num = ts - ts_list[i-1] + 1  # num need to interpolate and itself
                # print("i: {}, ts: {}, ts_list[i-1]: {}, num: {}".format(i, ts, ts_list[i-1], num))
                feature_ts_left = data[pid][staff][str(ts_list[i-1])][:-1]
                feature_ts_right = data[pid][staff][str(ts)][:-1]
                feat_tmp = []
                for j, (feat_l, feat_r) in enumerate(zip(feature_ts_left, feature_ts_right)):
                    feat = np.linspace(feat_l, feat_r, num)
                    # print("Interpolation: {}-{}: num: {}, feat: {}".format(feat_l, feat_r, num, feat))
                    feat_tmp.append(feat[1:])
                feat_tmp = np.array(feat_tmp).T
                for ii in range(feat_tmp.shape[0]):
                    feature_n.append(feat_tmp[ii].tolist())
            else:
                feature_n.append(data[pid][staff][str(ts)][:-1])
        
        # Step 2, Filling border 
        l_border_ts = cur_ts - diplacement_consistency_window
        r_border_ts = cur_ts + diplacement_consistency_window
        # print("l_border_ts: {}, r_border_ts: {}, len(feature_n): {}, feature_n:{}".format(l_border_ts, r_border_ts, len(feature_n), feature_n))
        l_ts = ts_list[0]
        r_ts = ts_list[-1]
        feature_all = []
        if l_border_ts < l_ts:
            for t in range(int(l_border_ts), int(l_ts)):
                # print("Filling left: {} use {}".format(t, feature_n[0]))
                feature_all.append(feature_n[0])
        feature_all.extend(feature_n)
        if r_border_ts > r_ts:
            for t in range(int(r_ts)+1, int(r_border_ts)):
                # print("Filling right: {} use {}".format(t, feature_n[-1]))
                feature_all.append(feature_n[-1])
        assert len(feature_all) == 2 * diplacement_consistency_window
        return feature_all

    def _get_video_path(self, bmk, pid, staff, ts, ts_list):
        """ Give a location, we can get the best view  by grid_views_map
        """
        xy = self._get_loc(pid, staff, ts)
        if not xy:
            return ""
        
        best_channel = get_best_camera_views(xy, self.grid_cameras_map)
        # print("xy: {}, best_channel: {}".format(xy, best_channel))
        if not best_channel:
            return ""
        else:
            channel = best_channel[0]
        video_name = ""
        h,m,s = ts_to_string(ts).split(":")
        video_start_time = "{}{:02}".format(h, int(m) - int(m) % 5)
        # print("min: {}, video_start_time: {}".format(m, video_start_time))
        # "/ssd/wphu/videos/GACNE_guangzhou_xhthwk/20210717/ch01004_20210717205000.mp4.cut.mp4"
        date = bmk.split('-')[-1]
        video_path_bmk = osp.join(self.args.video_path, '_'.join(bmk.split('-')))
        video_name = osp.join(video_path_bmk, "{}_{}{}00.mp4.cut.mp4".format(channel, date, video_start_time))
        # print("video_name:", video_name)
        return video_name

    def _get_loc(self, pid, staff, ts):
        """ Get loc by pid/staff/ts from new_xy.tar.gz
        """
        pid_car_inouts = self.debug_viz_infos[pid][staff]["pid_car_inouts"]
        staff_car_inouts = self.debug_viz_infos[pid][staff]["staff_car_inouts"]
        pid_locs_raw = self.pid_locs_raw
        
        if pid_car_inouts is not None:
            loc_pid = self._fill_missing_localization(pid_locs_raw[pid], pid_car_inouts)['loc']
            if ts in loc_pid:
                loc_cus = loc_pid[ts]
            else:
                loc_cus = []
        elif ts in pid_locs_raw[pid]['loc']:
            loc_cus = pid_locs_raw[pid]['loc'][ts].tolist()
        else:
            print("Customer: {} Lacking tracks at {}".format(pid, ts))
            loc_cus = []
        
        if staff_car_inouts is not None:
            loc_pid = self._fill_missing_localization(pid_locs_raw[staff], staff_car_inouts)['loc']
            if ts in loc_pid:
                loc_staff = loc_pid[ts]
            else:
                loc_staff = []
        elif ts in pid_locs_raw[staff]['loc'][ts]:
            loc_staff = pid_locs_raw[staff]['loc'][ts].tolist()
        else:
            print("Staff: {} Lacking tracks at {}".format(staff, ts))
            loc_staff = []
        
        loc_cus = np.array(loc_cus).tolist()
        loc_staff = np.array(loc_staff).tolist()
        try:
            if not loc_cus and not loc_staff:
                return []
            elif not loc_cus and loc_staff:
                loc_mean = loc_staff
            elif loc_cus and not loc_staff:
                loc_mean = loc_cus
            else:
                loc_mean = [(loc_cus[0] + loc_staff[0]) / 2, (loc_cus[1] + loc_staff[1]) / 2]
        except:
            import pdb; pdb.set_trace()
        return loc_mean
    

    def _fill_missing_localization(self, loc, inout_events):
        ret = dict(loc)
        for rid in inout_events:
            x, y = self.store.regions["CAR"][rid].exterior.coords.xy
            mx = np.mean(x)
            my = np.mean(y)
            for evt in inout_events[rid]:
                for t in range(int(evt["start_time"]), int(evt["end_time"])):
                    if t not in ret["loc"]:
                        ret["loc"][t] = np.array([mx, my])
                        ret["vel"][t] = 0.0
                        ret["acc"][t] = 0.0
        return ret

    def _get_debug_viz_infos(self, bmk):
        """ Load inout events from debug_viz_infos.json on this bmk
        """
        if self.debug_viz_infos_path is not None:
            file_path = os.path.join(self.debug_viz_infos_path, bmk, "debug_viz_infos.json")
            debug_viz_infos = json.loads(open(file_path, 'rb').read())
        return debug_viz_infos

    def __call__(self, bmk):
        return self.get_data_infos(bmk)


if __name__ == "__main__":
    bmks_train = ["VW-changchun-rq-20210728"]
    bmks_test = ["GACNE-guangzhou-xhthwk-20210717"]

    base_path = os.path.dirname(args.data_path)
    debug_viz_infos_path = os.path.join(base_path, 'viz_infos')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    datagen = DataGen(args, debug_viz_infos_path)
    for bmk in bmks_train + bmks_test:
        print("Processing bmk: {}".format(bmk))
        # Gen data_infos for each bmk
        data_infos = datagen(bmk)
        with open(os.path.join(args.save_path, 'data_infos_{}.json'.format(bmk)), 'w') as f:
            json.dump(data_infos, f, indent=2)

        


