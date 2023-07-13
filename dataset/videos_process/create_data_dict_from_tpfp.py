from copy import deepcopy
import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import logging


USE_COMBINE_FEATURE = False

def ts_to_string(ts, sep=":", sec_size=1):
    h = int(float(ts) / (sec_size * 60 * 60))
    m = int(float(ts) / (sec_size * 60)) % 60
    s = int(float(ts) / sec_size) % 60
    return "{0:02d}{3}{1:02d}{3}{2:02d}".format(h, m, s, sep)

class CreateDataset(object):
    def __init__(self, root, bmk, benchmark_path):
        self.root = root
        self.bmk = bmk
        self.tps = json.loads(open(osp.join(self.root, 'tpfp', bmk, 'tps.json'), 'rb').read())
        self.fps = json.loads(open(osp.join(self.root, 'tpfp', bmk, 'fps_remaining.json'), 'rb').read())
        self.viz_infos = json.loads(open(osp.join(self.root, 'viz_infos', bmk, 'debug_viz_infos.json'), 'rb').read())
        self.benchmark_path = benchmark_path
        self.match = json.loads(open(osp.join(root, 'match', bmk, 'match.json'), 'rb').read())

    @staticmethod
    def get_nooverlap_evts(pevt, gevt):
        """ Extract time which in pevt but not in gevt
        """
        evts = []
        if pevt[1] < gevt[0] or gevt[1] < pevt[0]:
            return evts
        
        if pevt[0] < gevt[0]:
            evts.append([pevt[0], gevt[0]])
        if gevt[1] < pevt[1]:
            evts.append([gevt[1], pevt[1]])
        return evts

    def get_fps_no_overlap_evt(self,):
        """ Only extract evt time which have no overlap with gt evt time
        """
        fps = []
        for fp in self.fps:
            ov = fp[4]
            if ov == 0.0:
                fps.append(fp)
                continue
            pevt = fp[5]
            gevts = fp[6]
            evts = []
            for gevt in gevts:
                evts.extend(self.get_nooverlap_evts(pevt, gevt))
            
            # split pred_evt into no-overlap evts
            for evt in evts:
                cur_fp = deepcopy(fp)
                cur_fp[5] = evt
                fps.append(cur_fp)
        return fps

    def get_feature(self, istp=True):
        """ read tps/fps and viz_infos, extract all infos to generate a feature vector
        """
        viz_infos = self.viz_infos
        # get positive data
        features = []
        fps = self.get_fps_no_overlap_evt()
        cases = self.tps if istp else fps
        for item in cases:
            pid, staff = item[0], item[2]
            evts = item[6] if istp else [item[5]]
            if pid not in viz_infos:
                print('pid: {} not found'.format(pid))
                continue
            elif staff not in viz_infos[pid]:
                print('staff {} not found in {}'.format(staff, viz_infos[pid]))
                continue
            elif 'ts_infos' not in viz_infos[pid][staff]:
                print('ts_infos not in viz_infos[pid][staff]')
                continue
            elif not viz_infos[pid][staff]['ts_infos']:
                print("viz_infos[pid][staff]['ts_infos'] is empty")
                continue
            
            # if pid not in viz_infos or staff not in viz_infos[pid] or \
            #     'ts_infos' not in viz_infos[pid][staff] or not viz_infos[pid][staff]['ts_infos']:
            #     continue
            
            ts_infos = viz_infos[pid][staff]['ts_infos']
            for evt in evts:
                for ts in range(int(evt[0]), int(evt[1]) + 1):
                    ts = str(ts)
                    if ts not in ts_infos:
                        continue
                    infos = ts_infos[ts]
                    interactions = {}
                    if 'interactions' not in infos:
                        for k in  'a_look_b','b_look_a','cofocal','view_align':
                            interactions[k] = 0.0
                        interactions['distance_3d'] = infos['dist'] if 'dist' in infos else 0.0
                    else:
                        interactions = infos['interactions']
                    
                    ts_info_empty = False
                    for i in 'dist', 'common_region_w', 'move_bonus', 'customer_incar', 'staff_incar':
                        if i not in infos:
                            ts_info_empty = True
                            break
                    if ts_info_empty:
                        continue
                    
                    discount = infos['score'] - infos['score-discount']
                    feature_vec = [infos['dist'], infos['common_region_w'], infos['move_bonus'], float(infos['customer_incar']), float(infos['staff_incar']), float(discount),
                                interactions['a_look_b'], interactions['b_look_a'], interactions['cofocal'], interactions['view_align'], interactions['distance_3d']]

                    if istp:
                        feature_label = feature_vec + [1]
                    else:
                        feature_label = feature_vec + [0]
                    features.append(feature_label)
        return features

    def get_feature_v2(self, istp=True):
        """ This version will generate pos/neg by score, score >= 0.1: pos; score < 0.1: neg
        read tps/fps and viz_infos, extract all infos to generate a feature vector
        """
        viz_infos = self.viz_infos
        # get positive data
        features_pos = []
        features_neg = []
        fps = self.get_fps_no_overlap_evt()
        cases = self.tps if istp else fps
        for item in cases:
            pid, staff = item[0], item[2]
            evts = item[6] if istp else [item[5]]
            if pid not in viz_infos:
                print('pid: {} not found'.format(pid))
                continue
            elif staff not in viz_infos[pid]:
                print('staff {} not found in {}'.format(staff, viz_infos[pid]))
                continue
            elif 'ts_infos' not in viz_infos[pid][staff]:
                print('ts_infos not in viz_infos[pid][staff]')
                continue
            elif not viz_infos[pid][staff]['ts_infos']:
                print("viz_infos[pid][staff]['ts_infos'] is empty")
                continue
            
            # if pid not in viz_infos or staff not in viz_infos[pid] or \
            #     'ts_infos' not in viz_infos[pid][staff] or not viz_infos[pid][staff]['ts_infos']:
            #     continue
            
            ts_infos = viz_infos[pid][staff]['ts_infos']
            for evt in evts:
                for ts in range(int(evt[0]), int(evt[1]) + 1):
                    ts = str(ts)
                    if ts not in ts_infos:
                        continue
                    infos = ts_infos[ts]
                    interactions = {}
                    if 'interactions' not in infos:
                        for k in  'a_look_b','b_look_a','cofocal','view_align':
                            interactions[k] = 0.0
                        interactions['distance_3d'] = infos['dist'] if 'dist' in infos else 0.0
                    else:
                        interactions = infos['interactions']
                    
                    ts_info_empty = False
                    for i in 'dist', 'common_region_w', 'move_bonus', 'customer_incar', 'staff_incar':
                        if i not in infos:
                            ts_info_empty = True
                            break
                    if ts_info_empty:
                        continue
                    
                    discount = infos['score'] - infos['score-discount']
                    feature_vec = [infos['dist'], infos['common_region_w'], infos['move_bonus'], float(infos['customer_incar']), float(infos['staff_incar']), float(discount),
                                interactions['a_look_b'], interactions['b_look_a'], interactions['cofocal'], interactions['view_align'], interactions['distance_3d']]

                    if infos['score'] >= 0.1:
                        feature_label = feature_vec + [1]
                        features_pos.append(feature_label)
                    else:
                        feature_label = feature_vec + [0]
                        features_neg.append(feature_label)
        return features_pos, features_neg

    def get_tp_fp_evts(self, ):
        """ Convert tp into dict[pid][staff]: set(evt_ts_list)
        """
        tp_evt_ts = {}
        ts_set = set()
        for item in self.tps:
            pid, staff = item[0], item[2]
            tp_evt_ts.setdefault(pid, {})
            gtevts = item[6]
            for evt in gtevts:
                ts_set |= set(range(int(evt[0]), int(evt[1]) + 1))
            tp_evt_ts[pid][staff] = ts_set
        # extract from fps
        for item in self.fps:
            pid, staff = item[0], item[2]
            tp_evt_ts.setdefault(pid, {})
            tp_evt_ts[pid].setdefault(staff, set())
            evts = item[6] if item[6] else [item[5]]
            for evt in evts:
                tp_evt_ts[pid][staff] |= set(range(int(evt[0]), int(evt[1]) + 1))
        return tp_evt_ts
  
    def get_feature_v3(self,):
        """ This version will generate pos/neg by score, score >= 0.1: pos; score < 0.1: neg
        read tps/fps and viz_infos, extract all infos to generate a feature vector
        """
        viz_infos = self.viz_infos
        # get positive data
        features_pos = []
        features_neg = []
        tpfp_evts = self.get_tp_fp_evts()
        for pid in viz_infos.keys():
            for staff in viz_infos[pid]:
                if not viz_infos[pid][staff]['ts_infos']:
                    continue
                ts_infos = viz_infos[pid][staff]['ts_infos']
                for ts in ts_infos:
                    # Exclude tp evts time that is consider as positive samples
                    if pid in tpfp_evts and staff in tpfp_evts[pid] and int(ts) in tpfp_evts[pid][staff]:
                        continue

                    if ts not in ts_infos:
                        continue
                    infos = ts_infos[ts]
                    interactions = {}
                    if 'interactions' not in infos:
                        for k in  'a_look_b','b_look_a','cofocal','view_align':
                            interactions[k] = 0.0
                        interactions['distance_3d'] = infos['dist'] if 'dist' in infos else 0.0
                    else:
                        interactions = infos['interactions']
                    
                    ts_info_empty = False
                    for i in 'dist', 'common_region_w', 'move_bonus', 'customer_incar', 'staff_incar':
                        if i not in infos:
                            ts_info_empty = True
                            break
                    if ts_info_empty:
                        continue
                    
                    discount = infos['score'] - infos['score-discount']
                    feature_vec = [infos['dist'], infos['common_region_w'], infos['move_bonus'], float(infos['customer_incar']), float(infos['staff_incar']), float(discount),
                                interactions['a_look_b'], interactions['b_look_a'], interactions['cofocal'], interactions['view_align'], interactions['distance_3d']]

                    if infos['score'] >= 0.1:
                        feature_label = feature_vec + [1]
                        features_pos.append(feature_label)
                    else:
                        feature_label = feature_vec + [0]
                        features_neg.append(feature_label)
        return features_pos, features_neg

    def get_gt_evts(self,):
        """ Loading all gt evts from GT file
        Return: dict[pid][staff]: set(evt_ts_list)
        """
        gtevts = {}
        brand, city, store, date = self.bmk.split('-')
        bmkpath = osp.join(self.benchmark_path, brand, city, '{}v7'.format(store), date, 'tpid_mappings.json')
        print("Loading gt file from: {}".format(bmkpath))
        gt = json.loads(open(bmkpath, 'rb').read())
        receptions = gt["reception"]
        int2str = gt["int2str"]

        # get pred_pid to gt_pid mappings
        match = self.match['match']
        pid_map = {}
        for k,v in match.items():
            pid_map[v] = k

        # Get all gt evts
        for pid_num in receptions.keys():
            gtpid = int2str[str(pid_num)]
            pred_pid = pid_map[gtpid] if gtpid in pid_map else None
            gtevts.setdefault(pred_pid, {})
            for evt in receptions[pid_num]:
                if evt['type'] != "reception":
                    continue
                gtstaff = str(evt['staff_pid'])
                staff = int2str[gtstaff] if gtstaff in int2str else None
                pred_staff = pid_map[staff] if staff in pid_map else None
                # print(pred_pid, pred_staff)
                gtevts[pred_pid].setdefault(pred_staff, set())
                gtevts[pred_pid][pred_staff] |= set(range(int(float(evt['video_start_time'])), int(float(evt['video_end_time'])) + 1))
        return gtevts

    
    def get_feature_v4(self, is_test=False):
        """ This version's definition:
        - Positive samples come from:
            - GT events time, Get rid of ts which dist is particularly large or ts which rule-based score < 0.1
        - Negtive samples come from:
            - Easy case: All common ts which not in GT evt time
            - Hard case: All fp evts time without overlap with any gtevts
        """
        viz_infos = self.viz_infos
        # get positive data
        features_pos = []
        features_neg = []
        features_pos_check = {}
        features_neg_check = {}

        features_neg_dirty = {}

        match = self.match['match']
        gt_evts = self.get_gt_evts()
        no_interaction_ts = 0
        neg_no_interaction_ts = 0
        total_ts = 0
        dirty_ts = 0
        both_dirty_ts = 0
        for pid in viz_infos.keys():
            if pid in match and match[pid] is None:
                continue
            for staff in viz_infos[pid]:
                if staff in match and match[staff] is None:
                    continue
                    
                if not viz_infos[pid][staff]['ts_infos']:
                    continue
                ts_infos = viz_infos[pid][staff]['ts_infos']
                for ts in ts_infos:
                    total_ts += 1
                    infos = ts_infos[ts]
                    interactions = {}
                    interactions_default = {'a_look_b': 0.9, 'b_look_a': 0.9, 'cofocal': 0.9, 'view_align': 3.0}
                    if 'interactions' not in infos:
                        no_interaction_ts += 1
                        # continue

                        for k in  interactions_default.keys():
                            interactions[k] =  interactions_default[k] if 'dist' in infos and infos['dist'] < 1.0 else 0.05
                        interactions['distance_3d'] = infos['dist'] if 'dist' in infos else 0.0
                    else:
                        interactions = infos['interactions']
                    
                    ts_info_empty = False
                    for i in 'dist', 'common_region_w', 'move_bonus', 'customer_incar', 'staff_incar':
                        if i not in infos:
                            ts_info_empty = True
                            break
                    if ts_info_empty:
                        continue
                    
                    discount = infos['score'] - infos['score-discount']
                    if USE_COMBINE_FEATURE:
                        cusincar_staffnotincar = 1.0 if infos['customer_incar'] and not infos['staff_incar'] else 0.0
                        look_each_other = interactions['a_look_b'] + interactions['b_look_a']
                        dist_diff = infos['dist'] - interactions['distance_3d']
                        very_close = [infos['dist'] < d*0.05 for d in range(0, 100, 5)]
                        both_incar = float(infos['customer_incar'] + infos['staff_incar'])
                        dist_a_look_b = (np.array(very_close) + interactions['a_look_b']).tolist()
                        dist_b_look_a = (np.array(very_close) + interactions['b_look_a']).tolist()
                        dist_cofocal = (np.array(very_close) + interactions['cofocal']).tolist()
                        feature_vec = [infos['dist'], infos['common_region_w'], infos['move_bonus'], float(infos['customer_incar']), float(infos['staff_incar']), float(discount), float(infos['max_dist_dloc']),
                                    cusincar_staffnotincar, look_each_other, dist_diff, *very_close, both_incar,
                                    interactions['a_look_b'], interactions['b_look_a'], interactions['cofocal'], interactions['view_align'], interactions['distance_3d']]
                    else:
                        feature_vec = [infos['dist'], infos['common_region_w'], infos['move_bonus'], float(infos['customer_incar']), float(infos['staff_incar']), float(discount), float(infos['max_dist_dloc']),
                                    interactions['a_look_b'], interactions['b_look_a'], interactions['cofocal'], interactions['view_align'], interactions['distance_3d']]
                    
                    
                    # Exclude tp evts time that is consider as positive samples
                    if pid in gt_evts and staff in gt_evts[pid] and int(ts) in gt_evts[pid][staff]:
                    # if infos['score'] >= 0.1:
                        # if not is_test and infos['score'] >= 0.1:
                        #     feature_label = feature_vec + [1]
                        #     features_pos.append(feature_label)
                        #     features_pos_check.setdefault(pid, {})
                        #     features_pos_check[pid].setdefault(staff, {})
                        #     features_pos_check[pid][staff][int(ts)] = feature_label
                        # elif is_test:
                            feature_label = feature_vec + [1]
                            features_pos.append(feature_label)
                            features_pos_check.setdefault(pid, {})
                            features_pos_check[pid].setdefault(staff, {})
                            features_pos_check[pid][staff][int(ts)] = feature_label
                    else:
                        if 'interactions' not in infos:
                            neg_no_interaction_ts += 1
                        
                        if infos['score'] >= 0.1:
                            feature_label = feature_vec + [0]
                            features_neg_dirty.setdefault(pid, {})
                            features_neg_dirty[pid].setdefault(staff, {})
                            features_neg_dirty[pid][staff][ts_to_string(ts)] = str(feature_label)
                            dirty_ts += 1
                            if 'interactions' not in infos:
                                both_dirty_ts += 1
                        # else:
                        feature_label = feature_vec + [0]
                        features_neg.append(feature_label)
                        features_neg_check.setdefault(pid, {})
                        features_neg_check[pid].setdefault(staff, {})
                        features_neg_check[pid][staff][int(ts)] = feature_label
        # sample negtive
        features_neg = np.array(features_neg)
        np.random.seed()
        np.random.shuffle(features_neg)
        # Note "// 8" means make sure pos/neg = 8:1
        # neg_length = len(features_pos) #// 8
        # features_neg = features_neg if is_test else features_neg[:neg_length]
        print(f"  | no_interaction_ts/total_ts: {no_interaction_ts}/{total_ts}={float(no_interaction_ts)/total_ts}")
        print(f"  | neg_no_interaction_ts/total_ts: {neg_no_interaction_ts}/{total_ts}={float(neg_no_interaction_ts)/total_ts}")
        print(f"  | dirty_ts/total_ts: {dirty_ts}/{total_ts}={float(dirty_ts)/total_ts}")
        print(f"  | (no_interaction_ts+ dirty_ts)/total_ts: {both_dirty_ts}/{total_ts}={float(both_dirty_ts)/total_ts}")
        return features_pos, features_neg.tolist(), \
            features_pos_check, features_neg_check, features_neg_dirty


    def dataset_statistic(self, bmk, features_pos, features_neg):
        """ Statistic distribution difference of tp/fp
        """
        root_viz = osp.join(self.root, 'statistics', bmk)
        if not osp.exists(root_viz):
            os.makedirs(root_viz)
        
        features_pos = np.array(features_pos[:-1])
        features_neg = np.array(features_neg[:-1])
        np.random.seed()
        np.random.shuffle(features_pos)
        np.random.seed()
        np.random.shuffle(features_neg)

        self.plot_feat(root_viz, features_pos.T, suffix="pos")
        self.plot_feat(root_viz, features_neg.T, suffix="neg")
        

    def plot_feat(self, root_viz, features, suffix, plot_length=10000):
        feat_map = ["dist", "common_region_w", "move_bonus", "customer_incar", "staff_incar", "discount",
                    "a_look_b", "b_look_a", "cofocal", "view_align", "distance_3d"]
        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'hotpink', 'greenyellow', 'darkslategray', 'darkblue', 'darkred']
        plt.figure()
        plt.title(f"statistic_{suffix}")
        for i, label in enumerate(feat_map[:6]):
            print("{}, Plotting {} {} ".format(i, suffix, label))
            feat = features[i][:min(plot_length, len(features[i]))]
            # feat = sorted(list(feat))
            X = range(len(feat))
            plt.scatter(X, feat, s=20, c=colors[i], marker='.', alpha=0.4, label=label)
            # for a,b in zip(X, feat):
            #     plt.text(a,b, '{}'.format(b), ha='center', va='bottom', fontsize=3)
            plt.legend()
        plt.savefig(f'{root_viz}/statistic_{suffix}.png', dpi=300)
        plt.close()


    def __call__(self, is_test=False):
        # feature_tp_pos = self.get_feature()  # Get samples from tps
        feature_fp_neg = self.get_feature(istp=False)  # Get samples from fps
        # features_pos, features_neg = self.get_feature_v2(istp=True)  # Get samples from tp evt time that rule-based, score >= 0 is pos; score < 0.1 is neg;
        # features_pos_commonts_v3, features_neg_commonts_v3 = self.get_feature_v3()  # Get samples from all commonts except for tp/fp evt time
        features_pos_commonts, features_neg_commonts, features_pos_check, features_neg_check, features_neg_dirty = self.get_feature_v4(is_test=is_test)  # Get samples by all GT events
        # return feature_tp_pos, feature_fp_neg
        # return feature_tp_pos, feature_fp_neg + features_neg # v4
        # return features_pos, feature_fp_neg + features_neg  # v5
        # return features_pos + features_pos_commonts_v3, feature_fp_neg + features_neg + features_neg_commonts_v3  # v6,v7
        return features_pos_commonts, feature_fp_neg + features_neg_commonts, features_pos_check, features_neg_check, features_neg_dirty  # v8


def compute_dirty_ratio(features_neg_check, features_neg_dirty):
    # get all common ts of negative samples
    all_common_ts = 0.0
    for pid, data in features_neg_check.items():
        for staff, feats in data.items():
            all_common_ts += len(feats.keys())
    
    # get common ts in dirty samples
    dirty_ts = 0.0
    for pid, data in features_neg_dirty.items():
        for staff, feats in data.items():
            dirty_ts += len(feats.keys())
    return float(dirty_ts) / all_common_ts

if __name__ == "__main__":
    """
    HONGQI-beijing-fkwd-20220109 data source: wphu@gpu016:/ssd/wphu/work_dir/HONGQI-beijing-fkwd-20220109-v7.5.0.0409
    "VOLVO-jinan-xfh-20210617"   data source: 
    """
    root = '/ssd/wphu/release.v7.5.0/Dataset/'
    benchmark_path = "/ssd/wphu/work_dir/Benchmarks_v7/"
    bmks = ["VOLVO-jinan-xfh-20210617", "VW-hefei-zl-20210727", "VW-tianjin-jz-20210727", "HONGQI-beijing-fkwd-20220109"]
    bmks_mini = ["VW-changchun-rq-20210728", "GACNE-guangzhou-xhthwk-20210717"]
    do_statistic = False

    # for bmk in bmks + bmks_mini:
    for bmk in ["GACNE-guangzhou-xhthwk-20210717"]:
        print(bmk)
        save_path = osp.join(root, 'data', bmk)
        if not osp.exists(save_path):
            os.makedirs(save_path)

        dataset = CreateDataset(root, bmk, benchmark_path)
        feature_pos, feature_neg, features_pos_check, features_neg_check, features_neg_dirty = dataset() if bmk != "GACNE-guangzhou-xhthwk-20210717" else dataset(is_test=False)

        # Plot statistic figure
        if do_statistic:
            dataset.dataset_statistic(bmk, feature_pos, feature_neg)

        # feature_pos = feature_pos[:len(feature_neg) * 5]
        data = feature_pos + feature_neg
        data_dict = {"feature_pos": features_pos_check, "feature_neg": features_neg_check}
        print("Total: {}, pos/neg: {}/{}, ratio: {}".format(len(data), len(feature_pos), len(feature_neg), len(feature_pos)/len(feature_neg)))

        ## Save all features into json
        with open(osp.join(save_path, 'data.json'), 'w')  as f:
            json.dump(data, f, indent=2)
        # Convert all features into dict
        with open(osp.join(save_path, 'data_dict.json'), 'w')  as f:
            json.dump(data_dict, f, indent=2)
        print("Success! Dataset has been saved into {}".format(save_path))

        with open(osp.join(save_path, 'features_neg_dirty.json'), 'w') as f:
            json.dump(features_neg_dirty, f, indent=2)

        # compute dirty ratio
        dirty_ratio = compute_dirty_ratio(features_neg_check, features_neg_dirty)
        print(f"dirty ratio(rule-based score>=0.1 in negative samples): {round(dirty_ratio * 100, 2)}%")

