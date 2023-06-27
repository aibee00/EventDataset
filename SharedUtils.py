import bisect
from bisect import bisect_left, bisect_right
from scipy.spatial.distance import cdist
import cv2
import cmath
import os
from copy import deepcopy
from collections import defaultdict
import numpy as np
from scipy.signal import convolve
from shapely.geometry.polygon import Polygon
import logging

#将上级目录添加到工作路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))

from proto import track3d_pb2, person_tracking_result_pb2


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

CARDINAL_SEP = ";;;"


def sec2str(second):
    second = int(second)
    return "%02d:%02d:%02d" % (second / 3600, (second % 3600) / 60, second % 60)


def g_pid(v):
    return v.pid()


def g_stime(v):
    return v._start_time


def g_etime(v):
    return v._end_time


def g_duration(v):
    return v._end_time - v._start_time


def g_cid(v):
    return v.rid()


def g_uniqkey(v):
    return f"{g_pid(v)}-{g_cid(v)}-{g_sid(v)}"


def g_sid(v):
    return v.sid()


def timeIoU(x0, x1, y0, y1):
    a = min(x1, y1) - max(x0, y0)
    xx = x1 - x0
    yy = y1 - y0
    return max([float(a) / (xx + yy - a), 0])


def timeIoUmin(x0, x1, y0, y1):
    a = min(x1, y1) - max(x0, y0)
    xx = x1 - x0
    yy = y1 - y0
    return max([float(a) / min(xx, yy), 0])


def tsf2sec(ss):
    if ':' in ss and len(ss) > len("HH:MM:SS"):
        # assume we have a format as "YY-MM-DD HH:mm:ss"
        ss = ss.split(' ')[-1]
        pp = ss.split(':')
    elif len(ss) == len("YYYYMMDDHHMMSS"):
        pp = [ss[8:10], ss[10:12], ss[12:14]]
    else:
        pp = ss.split(':')
    hh = int(pp[0])
    if (ss[-2:] == 'PM'):
        hh += 12
    return hh * 3600 + int(pp[1]) * 60 + int(pp[2].split()[0])

def str2ts(str, sec_size=1, sep=":"):
    if sep:
        h,m,s=str.split(sep)
    else:
        h = str[0:2]
        m = str[2:4]
        s = str[4:6]
    return int(h)*3600 + int(m)*60 + int(s)

def convert_unix_time_to_ts(unix_time, gmt=8):
    # gmt=8: beijing time
    ts = (unix_time / 1000 + gmt*3600) % 86400
    return int(ts)


def timestamp(ts_, enforce24hr=True):
    ts = int(ts_)
    if enforce24hr:
        ts = min([ts, 3600 * 24 - 1])
    return ('{}:{}:{}'.format(str(ts // 3600).zfill(2),
                              str((ts % 3600) // 60).zfill(2),
                              str(ts % 60).zfill(2)))


def hms_string_to_timestamp(hms_string):
    assert len(hms_string) == 6
    h = int(hms_string[:2])
    m = int(hms_string[2:4])
    s = int(hms_string[4:6])
    return h * 3600 + m * 60 + s

def hms2sec(hms):
    h = int(hms / 10000)
    m = int(hms / 100) % 100
    s = hms % 100
    return h * 3600 + m * 60 + s

class CircularStatsCalulator(object):
    # https://gist.github.com/kn1cht/89dc4f877a90ab3de4ddef84ad91124e
    # https://en.wikipedia.org/wiki/Circular_mean
    def __init__(self):
        return

    def mean(self, angles, deg=True):
        #Circular mean of angle data(default to degree)
        a = np.deg2rad(angles) if deg else np.array(angles)
        angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
        mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
        return round(np.rad2deg(mean) if deg else mean, 7)

    def var(self, angles, deg=True):
        '''Circular variance of angle data(default to degree)
        0 <= var <= 1
        '''
        a = np.deg2rad(angles) if deg else np.array(angles)
        angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
        r =abs(angles_complex.sum()) / len(angles)
        return round(1 - r, 4)

    def std(self, angles, deg=True):
        '''Circular standard deviation of angle data(default to degree)
        0 <= std
        '''
        a = np.deg2rad(angles) if deg else np.array(angles)
        angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
        r = abs(angles_complex.sum()) / len(angles)
        std = np.sqrt(-2 * np.log(r))
        return round(np.rad2deg(std) if deg else std, 4)

class TrackWrapper(object):
    def __init__(self):
        self._pid = None
        self._ts_vec = None
        self._pts_vec = None
        self._sit_vec = None
        self._head_angles = None
        self._height = None
        self._interaction_dict = None
        self._body_features = None
        self._sight_sectors = {}
        self._head_std = {}
        self._store = None

    def appearance_distance(self, other, p=25):
        if self.body_features is not None and other.body_features is not None:
            dists = 1.0 - np.matmul(self.body_features, other.body_features.T)
            return np.percentile(dists.flatten(), p)
        else:
            # large distance
            return 1.0

    @property
    def body_features(self):
        return self._body_features

    def load_appearance(self, fname):
        tracks = person_tracking_result_pb2.Track()
        with open(fname, 'rb') as file_pointer:
            tracks.ParseFromString(file_pointer.read())
        for trk in tracks.single_view_tracks:
            features = []
            for feat in trk.body_feature.features:
                features.append([v for v in feat.feats])
            if not features:
                continue
            # normalize
            features = np.array(features)
            features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
            ft = np.mean(features, axis=0)
            if self._body_features is None:
                self._body_features = [ft.tolist()]
            else:
                self._body_features.append(ft.tolist())
        if self._body_features is not None:
            self._body_features = np.array(self._body_features)
            print(f"append {self._body_features.shape[0]} number of body features from {fname}")

    @classmethod
    def from_pb_file(tkr, fpath, store = None):
        tkr = TrackWrapper()
        tracks = track3d_pb2.Tracks()
        with open(fpath, 'rb') as f:
            tracks.ParseFromString(f.read())
        # Should only be a single track in the pb file
        track = tracks.tracks[0]
        tkr._pid = str(track.pid_str)
        tkr.ts_vec = [f.timestamp for f in track.frames]
        tkr.pts_vec = \
            [(f.nose_pixels.x, f.nose_pixels.y) for f in track.frames]
        tkr.sit_vec = [f.detection_posture_type for f in track.frames]
        tkr.head_angles = [f.head_angle if not np.isnan(f.head_angle) else 0 for f in track.frames]
        tkr.height = [f.height_estimate for f in track.frames]
        tkr._store = store
        tkr._cached_scores = {} # cached scores from various steps
        tkr.sort()
        tkr.get_movement()
        return tkr
    
    @property
    def attributes(self):
        assert self._store
        assert self._pid in self._store.pid_attributes
        return self._store.pid_attributes[self._pid]

    def update_from_trk(self, trk):
        # update features at each ts from trk that self doesn't have
        assert self._pid == trk.pid
        for ts in trk.ts_vec:
            if ts in self.ts_vec:
                continue
            ts_idx = bisect_left(self.ts_vec, ts)
            ts_idx_trk = bisect_left(trk.ts_vec, ts)
            self._pts_vec = np.insert(self._pts_vec, ts_idx, trk.pts_vec[ts_idx_trk], axis=0)
            self._vel_vec = np.insert(self._vel_vec, ts_idx, trk.vel_vec[ts_idx_trk])
            self._acc_vec = np.insert(self._acc_vec, ts_idx, trk.acc_vec[ts_idx_trk])
            self._ts_vec = np.insert(self._ts_vec, ts_idx, trk.ts_vec[ts_idx_trk])

    def update_person(self, store):
        CircularCalulator = CircularStatsCalulator()
        self._update_sight_sector(store)
        self._update_head_std(CircularCalulator)
        self.scaled_vel = []
        for i in self.vel_vec:
            self.scaled_vel.append(1.0 * i / store.meter2pixel)

    def _update_head_std(self, calculator):
        max_gap = 5
        for i, ts in enumerate(self.ts_vec):
            head_bearing = self.head_angles[i]
            if head_bearing in [0,360]:
                continue
            pre_ts_i = max( 0, i - 1)
            next_ts_i = min(len(self.ts_vec) - 1, i + 1)
            if self.ts_vec[next_ts_i] - self.ts_vec[pre_ts_i] > max_gap:
                # don't collect head poses from large gaps
                looking_bearings = [self.head_angles[i]]
            else:
                looking_bearings = [
                    self.head_angles[j] for j in [pre_ts_i, i, next_ts_i]
                    ]
            self._head_std[ts] = calculator.std(looking_bearings)
        return

    def _update_sight_sector(self, store):
        sight_distance = 3 # meter
        sight_pixels = sight_distance * store.meter2pixel
        fov_angle = 35
        for ts, loc, head_bearing in zip(self.ts_vec, self.pts_vec, self.head_angles):
            if head_bearing in [0,360]:
                continue
            inner_sector = [loc]
            outter_sector = []
            inner_coords = []
            # fov_sector_coords = [loc]
            angle_rad = head_bearing * np.pi / 180.0
            for theta in np.linspace(
                angle_rad - fov_angle * np.pi / 180,
                angle_rad + fov_angle * np.pi / 180,
                10):
                outter_pt = [
                    loc[0] + sight_pixels * np.cos(theta),
                    loc[1] + sight_pixels * np.sin(theta)
                    ]
                inner_coords.append([
                    loc[0] + 0.5 * sight_pixels * np.cos(theta),
                    loc[1] + 0.5 * sight_pixels * np.sin(theta)
                    ])
                outter_sector.append(outter_pt)
            inner_sector.extend(inner_coords)
            inner_sector.append(loc)
            outter_sector.extend(inner_coords[::-1])
            outter_sector.append(outter_sector[0])
            outter_sector_polygon = Polygon(outter_sector)
            inner_sector_polygon = Polygon(inner_sector)
            self._sight_sectors[ts] = {"inner": inner_sector_polygon, "outter": outter_sector_polygon}
        return

    @property
    def sight_sectors(self):
        return self._sight_sectors
    
    @sight_sectors.setter
    def sight_sectors(self, value):
        self._sight_sectors = value

    @property
    def head_std(self):
        return self._head_std

    @head_std.setter
    def head_std(self, value):
        self._head_std = value

    @property
    def is_child(self):
        CHILD_LS = [1, 2, 3]
        assert self._store, "store object with pid attribute is required for detemining is child"
        standing_height = [
            height
            for posture, height in zip(self.sit_vec, self.height)
            if posture != track3d_pb2.DetectionPostureType.DETECTION_POSTURE_SEATED and np.abs(height - 1.6) > 1e-3
        ]
        if not standing_height:
            return False
        total = 0.0
        for ls in CHILD_LS:
            if ls in self.attributes['life_stage']:
                total += self.attributes['life_stage'][ls]
        p = max(95, int(float(len(standing_height) - 25) / len(standing_height) * 100))
        p95_height = np.percentile(standing_height, p)
        if (p95_height < 1.6 and total >= 0.5) or total >= 0.9:
            ischild = True
        else:
            ischild = False
        # dbgoutput = "/ssd/kgli/LIXIANG_error_analysis/anlysis/{}/child_dbg_{}/".format(ischild)
        # dbgstore = "LIXIANG_beijing_hsh_20220709"
        # imgdbg_name = "{}_{}_{}.jpg".format(np.round(total, 3), np.round(p95_height, 3), self._pid)
        # img = glob(
        #         os.path.join(
        #             '/ssd/kgli/LIXIANG_error_analysis/anlysis/{}/input/viz'.format(dbgstore),
        #             "{}*.jpg".format(self._pid)
        # #         ))[0]
        # if not os.path.exists(dbgoutput):
        #     os.mkdir(dbgoutput)
        # shutil.copy2(img, osp.join(dbgoutput, imgdbg_name))
        return ischild

    @property
    def cached_scores(self):
        return self._cached_scores

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, data):
        self._height = data

    @property
    def pid(self):
        return self._pid

    @pid.setter
    def pid(self, pid):
        self._pid = pid

    @property
    def ts_vec(self):
        return self._ts_vec

    @ts_vec.setter
    def ts_vec(self, val):
        self._ts_vec = val

    @property
    def pts_vec(self):
        return self._pts_vec

    @pts_vec.setter
    def pts_vec(self, val):
        self._pts_vec = val

    @property
    def sit_vec(self):
        return self._sit_vec

    @sit_vec.setter
    def sit_vec(self, val):
        self._sit_vec = val

    @property
    def vel_vec(self):
        return self._vel_vec

    @property
    def acc_vec(self):
        return self._acc_vec

    @property
    def head_angles(self):
        return self._head_angles

    @head_angles.setter
    def head_angles(self, val):
        self._head_angles = val

    @property
    def interaction_dict(self):
        return self._interaction_dict

    @interaction_dict.setter
    def interaction_dict(self, val):
        self._interaction_dict = val

    def get_movement(self):
        '''
        calculate the velocity and acceleration
        '''
        N = 5
        window = np.ones((2 * N + 1, 1)) / (2 * N + 1)
        segs = self.ts_segments()
        vel = np.zeros(self._pts_vec.shape, dtype=np.float32)
        acc = np.zeros(self._pts_vec.shape, dtype=np.float32)
        if len(self._pts_vec) == 0:
            return
        for idx in segs:
            pts = self._pts_vec[idx]
            vel[idx[1:]] = pts[1:] - pts[:-1]
            # / (self._ts_vec[idx][1:] - self._ts_vec[idx][:-1])
            vel[idx] = convolve(vel[idx], window, mode='same')
            acc[idx[1:]] = vel[idx[1:]] - vel[idx[:-1]]
            acc[idx] = convolve(acc[idx], window, mode='same')
        self._vel_vec = np.linalg.norm(vel, axis=1)
        self._acc_vec = np.linalg.norm(acc, axis=1)
        if self._store:
            self.update_person(self._store)

    def sub_time_region(self, time_region_gap=10):
        all_ts = np.array(self._ts_vec, dtype=np.int32)
        
        all_ts = np.sort(all_ts)
        index = np.where(all_ts[1:] - all_ts[:-1] > time_region_gap)[0] + 1
        sub_region_ts = np.split(all_ts, index)

        regions = []
        for ts in sub_region_ts:
            regions.append(
                [min(ts), max(ts)]
            )
        return regions
    
    def show_sub_time_regions(self):
        sub_regions = self.sub_time_region()
        for i, region in enumerate(sub_regions):
            mlog.info("Track {} 's {}th region is ({} --> {})".format(self.pid, i, sec2str(region[0]), sec2str(region[1])))
    
    def get_heights_in_time_region(self, st, ed):
        lidx = bisect_left(self.ts_vec, st)
        ridx = bisect_right(self.ts_vec, ed)
        return self.height[lidx: ridx]
    
    def ts_segments(self, timeout=5):
        '''
        Get segmented indices of time with timeout
        '''
        ret = [[]]
        for i, t in enumerate(self._ts_vec):
            if not ret[-1]:
                ret[-1].append(i)
            else:
                if self._ts_vec[ret[-1][-1]] < t - timeout:
                    ret.append([i])
                else:
                    ret[-1].append(i)
        return ret

    def txy(self, time_range=None):
        if time_range is None:
            return zip(self._ts_vec, self._pts_vec)
        else:
            st_idx = bisect.bisect_left(self._ts_vec, time_range[0])
            end_idx = bisect.bisect_right(self._ts_vec, time_range[1])
            if len(self._ts_vec[st_idx:end_idx]):
                return zip(self._ts_vec[st_idx:end_idx],
                           self._pts_vec[st_idx:end_idx])
            else:
                return []

    def txyhead(self, time_range=None):
        if time_range is None:
            return zip(self._ts_vec, self._pts_vec, self._head_angles)
        else:
            st_idx = bisect.bisect_left(self._ts_vec, time_range[0])
            end_idx = bisect.bisect_right(self._ts_vec, time_range[1])
            if len(self._ts_vec[st_idx:end_idx]):
                return zip(self._ts_vec[st_idx:end_idx],
                           self._pts_vec[st_idx:end_idx],
                           self._head_angles[st_idx:end_idx])
            else:
                return []

    def timed_data(self, time_range=None):
        if time_range is None:
            return zip(self._ts_vec, self._pts_vec,
                       self._head_angles, self._height,
                       self._sit_vec)
        else:
            st_idx = bisect.bisect_left(self._ts_vec, time_range[0])
            end_idx = bisect.bisect_right(self._ts_vec, time_range[1])
            if len(self._ts_vec[st_idx:end_idx]):
                return zip(self._ts_vec[st_idx:end_idx],
                           self._pts_vec[st_idx:end_idx],
                           self._head_angles[st_idx:end_idx],
                           self._height,
                           self._sit_vec)
            else:
                return []

    def sort(self):
        ts_idx = np.argsort(self._ts_vec)
        self._ts_vec = np.array([self._ts_vec[i] for i in ts_idx])
        self._pts_vec = \
            np.array([self._pts_vec[i] for i in ts_idx])
        self._sit_vec = [self._sit_vec[i] for i in ts_idx]
        self._head_angles = \
            np.array([self._head_angles[i] for i in ts_idx])
        self._height = \
            [self._height[i] for i in ts_idx]

    def from_json_track(self, person):
        self._pid = person["pid"]
        positions = person["position"]
        self._ts_vec = []
        self._pts_vec = []
        self._sit_vec = []
        for p in positions:
            self._ts_vec.append(p['timestamp'])
            self._pts_vec.append((p['coordinates'][0], p['coordinates'][1]))
            try:
                self._sit_vec.append(p['detection_posture_type'])
            except KeyError:
                self._sit_vec.append(0)

    def compute_avg_loc(self, idx, st, et, debug=False):
        s_i = bisect.bisect_left(self._ts_vec, st, 0, len(self._ts_vec))
        e_i = bisect.bisect_left(self._ts_vec, et + 1, 0, len(self._ts_vec))
        # if debug:
        #     print(f"averge s loc: [{self._ts_vec[s_i]}~{self._ts_vec[idx]}], pre one {self._ts_vec[s_i - 1]}")
        #     print(f"averge e loc: [{self._ts_vec[idx + 1]}~{self._ts_vec[e_i - 1]}], next one {self._ts_vec[e_i]}")
        avg_s_loc = np.mean(self._pts_vec[s_i:(idx + 1)], axis=0)
        avg_e_loc = np.mean(self._pts_vec[(idx + 1):e_i], axis=0)
        return avg_s_loc, avg_e_loc

    def fill_gap_by_region_inout(self, region_inout_evt, window=2, fill_step=1, debug=False):
        self.sort()
        self.get_movement()
        self._ts_vec = self._ts_vec.tolist()
        if debug:
            import os
            local_score_folder = "/tmp/vw_poc_debug_pid_track/"
            if not os.path.exists(local_score_folder):
                os.makedirs(local_score_folder)
            region_inout_se = []
            local_score_json_path = os.path.join(local_score_folder, "org_%s_tracks_info.npz" % self._pid)
            np.savez(local_score_json_path,
                     ts=self._ts_vec,
                     pts=self._pts_vec,
                     vel=self._vel_vec,
                     acc=self._acc_vec
                     )
        for evt in region_inout_evt:
            i = bisect.bisect_left(self._ts_vec, g_stime(evt) + 1, 0, len(self._ts_vec))
            i -= 1
            # compute average score in window
            if i == len(self._ts_vec) - 1:
                continue
            avg_s_loc, avg_e_loc = self.compute_avg_loc(i, g_stime(evt) - window + 1, g_etime(evt) + window - 1,
                                                        debug=debug)
            if debug:
                region_inout_se.append([g_stime(evt), g_etime(evt)])
                print(f"Find gap between {sec2str(g_stime(evt))} and {sec2str(g_etime(evt))}")
                print(f"Insert between {self._ts_vec[i]} and {self._ts_vec[i + 1]}")
                print(f"Average loc before {avg_s_loc} and Average loc after {avg_e_loc}")
            slope = (avg_e_loc - avg_s_loc) / (int(self._ts_vec[i + 1]) - int(self._ts_vec[i]))
            new_vel = []
            new_acc = []
            new_ts = list(range(int(self._ts_vec[i]) + fill_step, int(self._ts_vec[i + 1]), fill_step))
            new_pts = np.zeros((len(new_ts), 2), dtype=np.float32)
            for j in range(len(new_ts)):
                new_pts[j] = self._pts_vec[i] + (new_ts[j] - self._ts_vec[i]) * slope
                new_vel.append(np.linalg.norm(slope) * 1.0)
                new_acc.append(0.0)
            self._pts_vec = np.insert(self._pts_vec, i + 1, new_pts, axis=0)
            self._vel_vec = np.insert(self._vel_vec, i + 1, new_vel)
            self._acc_vec = np.insert(self._acc_vec, i + 1, new_acc)
            self._ts_vec = np.insert(self._ts_vec, i + 1, new_ts)
        if debug:
            local_score_json_path = os.path.join(local_score_folder, "%s_tracks_info.npz" % self._pid)
            np.savez(local_score_json_path,
                     region_inout_se=region_inout_se,
                     ts=self._ts_vec,
                     pts=self._pts_vec,
                     vel=self._vel_vec,
                     acc=self._acc_vec
                     )
    def get_fps(self):
        fps_dict = defaultdict(int)
        for ts in self._ts_vec:
            fps_dict[int(ts)] += 1
        return np.mean(list(fps_dict.values()))

    def percentile_height(self, percentile = 95):
        return np.percentile(self.height, percentile)

    def is_sitting(self, ts, sample_range = 30, mode = 'strict'):
        '''
        use height to estimate whether the person is sitting or not
        '''
        threshold = 0.15
        left = bisect_right(self.ts_vec, ts)
        right = bisect_right(self.ts_vec, ts + sample_range)
        person_height = self.percentile_height()
        # if person_height < 1.45: # add this could improve precision but would filter some obvious cases
        #     return True
        if mode == 'strict':
            cur_height = np.median([self.height[i] for i in range(left, right)])
        elif mode == 'relax':
            cur_height = min([self.height[i] for i in range(left, right)])
        return (person_height - cur_height) / person_height > threshold

    def is_store_inout(self, t1, t2, min_dist = 2.5, sample_range = 10, min_gap = 25, check_store = False, percentile = 50):
        '''
        determine whether the person exited from store boundardy at t1 and returned at t2
        (1) dist to store boundary is small at t1/t2
        (2) velocity > min_v, before t1 and after t2
        '''
        if t2 - t1 < min_gap:
            return False
        static_t1 = self.is_static(t1, sample_range, min_dist, percentile)
        static_t2 = self.is_static(t2, sample_range, min_dist, percentile)
        if static_t1 or static_t2:
            mlog.info("{} not found t1 {} static {} t2 {} static {}".format(
                self.pid, sec2str(t1), static_t1, sec2str(t2), static_t2))
            return False
        t1_i = bisect_left(self.ts_vec, t1)
        t2_i = bisect_left(self.ts_vec, t2)
        if 'EFFECTIVE_REGION' in self._store.fast_distance2region:
            store_regions = self._store.fast_distance2region['EFFECTIVE_REGION']
        elif check_store:
            store_regions = self._store.fast_distance2region['STORE']
        else:
            return False
            # store_regions = self._store.fast_distance2region['STORE']
        pt_1 = self.pts_vec[t1_i]
        pt_2 = self.pts_vec[t2_i]
        dist_1 = []
        dist_2 = []
        for reg_id, region in store_regions.items():
            dist_1.append(region.distance(pt_1) / self._store.meter2pixel)
            dist_2.append(region.distance(pt_2) / self._store.meter2pixel)
        dist_1, dist_2 = max(dist_1), max(dist_2)
        if dist_1 > -min_dist and dist_2 > -min_dist:
            mlog.info("{} found: t1: {} t2 {} dist1 {} dist2 {}".format(self.pid, sec2str(t1), sec2str(t2), dist_1, dist_2))
            # self.viz(t1-5, t2+5, '/ssd/kgli/dbginoutfp', self._store, region_coords=region.original_coords)
            # self.viz(tsf2sec("10:56:37"), tsf2sec("11:03:00"), '/ssd/kgli/dbginoutfp', self._store, region_coords=region.original_coords)
            return True
        else:
            mlog.info("{} not found t1: {} t2 {} dist1 {} dist2 {}".format(self.pid, sec2str(t1), sec2str(t2), dist_1, dist_2))
            return False

    def is_static_during(self, st, et, static_dist = 0.8, percentile = 75):
        '''
        Use height to estimate whether the person is sitting or not
        '''
        self.cached_scores.setdefault('static_during', {})
        cur_key = "{}_{}_{}_{}".format(st, et, static_dist, percentile)
        if cur_key not in self.cached_scores['static_during']:    
            assert self._store
            ts_vec = self.ts_vec
            xy_vec = self.pts_vec
            tset = set()
            pts = []
            i = bisect_left(ts_vec, st)
            j = bisect_right(ts_vec, et)
            for idx in range(i, j):
                t = int(ts_vec[idx])
                tset.add(t)
                pts.append(xy_vec[idx])
            dists = cdist(pts, pts[::-1]) / self._store.meter2pixel
            self.cached_scores['static_during'][cur_key] =np.percentile(dists, percentile) < static_dist
        return self.cached_scores['static_during'][cur_key]

    def is_static(self, ts, sample_range = 5, static_dist = 0.8, percentile = 50):
        '''
        Use height to estimate whether the person is sitting or not
        '''
        assert self._store
        ts_vec = self.ts_vec
        xy_vec = self.pts_vec
        tset = set()
        pts = []
        i = bisect_left(ts_vec, ts - sample_range)
        j = bisect_right(ts_vec, ts + sample_range)
        for idx in range(i, j):
            t = int(ts_vec[idx])
            tset.add(t)
            pts.append(xy_vec[idx])
        dists = cdist(pts[:sample_range], pts[-sample_range:]) / self._store.meter2pixel
        return np.percentile(dists, percentile) < static_dist

    def get_features_from_t(self, t):
        if isinstance(t, str):
            t = tsf2sec(t)
        if t not in self.ts_vec:
            print("t: {} not in {}".format(sec2str(t), self._pid))
            return {}
        idx = self.ts_vec.tolist().index(t)
        if t in self.head_std:
            head_std = self.head_std[t]
        else:
            head_std = None
        if self._store:
            scaled_v = self.vel_vec[idx]/self._store.meter2pixel
        else:
            scaled_v = None
        return {
            'xy': self.pts_vec[idx],
            'height': self.height[idx],
            'vel': self.vel_vec[idx],
            'scaled_v': scaled_v,
            'acc': self.acc_vec[idx],
            'head_std': head_std
        }

    def _get_region_scores(self, ts, region_id, region_type, inout_only = False, region_score_cache_key = 'region_visit_scores'):
        score_info = self.cached_scores[region_score_cache_key][region_id][ts]
        res = []
        for k in sorted(score_info.keys()):
            if inout_only and 'inout' not in k:
                continue
            v = score_info[k]
            res.append("{}:{}".format(
                str(k), str(np.round(v, 2))
                ))
        return '_'.join(res)

    def viz(self, st, et, output_path, debug_region = None, region_coords = None, post_fix = '', \
        region_visit_score_key = 'head_pose_aware_region_scores', region_score_cache_key = 'region_visit_scores'
        ):
        store = self._store
        if isinstance(st, str):
            st, et = tsf2sec(st), tsf2sec(et)
        def _draw_counter(pts, floor_img, color, line_width=4):
            pts = [[int(i), int(j)] for i,j in pts]
            for j, pt in enumerate(pts):
                pt = tuple(pt)
                cv2.circle(floor_img, pt, 10, color, -1)
                if j < len(pts) - 1:
                    pt2 = pts[j + 1]
                    pt2 = tuple(pt2)
                    cv2.line(floor_img, pt, pt2, color, line_width)
                else:
                    pt2 = pts[0]
                    pt2 = tuple(pt2)
                    cv2.line(floor_img, pt, pt2, color, line_width)
        output_path = '{}/{}'.format(output_path, self.pid)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("output to: {}".format(output_path))
        median_height = np.round(self.percentile_height(), 2)
        floor_img = cv2.imread(store.floormap_path)
        pt_color = (255, 255, 0)
        base_color = (0, 255, 0)
        region_color = (255, 0, 0)
        line_color =(0, 0, 255)
        dist = 2.5
        dist_pixel = int(dist * store.meter2pixel)
        if region_coords:
            _draw_counter(region_coords, floor_img, region_color, 4)
        # At each ts from st to et
        #   visualize: (1) xy, head_angle (2) region_coords
        #   output name contains: current_height, median_height, v, acc, sit, post_fix
        for ts in range(int(st), int(et) + 1, 1):
            if ts not in self._ts_vec:
                continue
            cur_floor_img = deepcopy(floor_img)
            ts_string = sec2str(ts)
            print("vis: {}".format(ts_string))
            idx = bisect.bisect_left(self._ts_vec, ts)
            cur_height = np.round(self._height[idx], 2)
            cur_sit = self._sit_vec[idx]
            cur_v = np.round(self._vel_vec[idx] / store._meter2pixel, 2)
            pt = tuple([int(i) for i in self._pts_vec[idx].tolist()])
            cv2.circle(cur_floor_img, pt, 20, pt_color, -1)
            cur_head_angle = self.head_angles[idx]
            if cur_head_angle != 0 and cur_head_angle != 360:
                angle = cur_head_angle
                angle_rad = cur_head_angle * np.pi / 180.0
                pt_1 = (int(pt[0]), int(pt[1]))
                pt_2 = (
                    int(pt[0] + dist_pixel * np.cos(angle_rad)),
                    int(pt[1] + dist_pixel * np.sin(angle_rad))
                    )
                cv2.line(cur_floor_img, pt_1, pt_2, line_color, 8)
                _draw_counter(self.sight_sectors[ts]['inner'].exterior.coords, cur_floor_img, pt_color, 2)
                _draw_counter(self.sight_sectors[ts]['outter'].exterior.coords, cur_floor_img, pt_color, 2)
                pose_std = np.round(self.head_std[ts], 2)
                if region_visit_score_key in self.cached_scores:
                    head_pose_candidate = self.cached_scores[region_visit_score_key]['region_candidates'][ts]
                    if head_pose_candidate:
                        for k,v in head_pose_candidate.items():
                            reg_id, reg_type = k.split(CARDINAL_SEP)
                            candidate = "{}:{}".format(reg_id, reg_type)
                            region = self._store._carevent_distance2region[reg_type][reg_id]
                            coords = region.original_coords
                            _draw_counter(coords, cur_floor_img, line_color, 2)
                    else:
                        candidate = "No:Candidate"
            else:
                candidate = pose_std = angle_rad = angle = -1
            if debug_region:
                if CARDINAL_SEP in debug_region:
                    reg_id, reg_type = debug_region.split(CARDINAL_SEP)
                else:
                    reg_id, reg_type = debug_region, 'CAR'
                    debug_region = CARDINAL_SEP.join([reg_id, reg_type])
                region_debug = self._store._carevent_distance2region[reg_type][reg_id]
                
                dbgcoords = region_debug.exterior.coords
                debug_region_path = os.path.join(output_path, debug_region)
                if not os.path.exists(debug_region_path):
                    os.makedirs(debug_region_path)
                region_score_info = self._get_region_scores(ts, debug_region, reg_type)
                _draw_counter(dbgcoords, cur_floor_img, base_color, 4)
                post_fix_temp = str(candidate) + '_' + region_score_info + '_' +post_fix
                img_path = debug_region_path
            else:
                post_fix_temp = str(candidate) + '_' + post_fix
                img_path = output_path
            img_name = os.path.join(
                img_path,
                "{}_m-h:{}_c-h:{}_v:{}_pose-std:{}_{}.jpg".format(
                    ts_string, median_height, cur_height, cur_v, pose_std, post_fix_temp
                    )
                )
            print("save: {}".format(img_name))
            down_width = int(cur_floor_img.shape[0]/2)
            down_height = int(cur_floor_img.shape[1]/2)
            down_points = (down_height, down_width)
            cv2.imwrite(img_name, cv2.resize(cur_floor_img, down_points))



class TrackWrapperWithoutStore(TrackWrapper):
    """ This is a overwrited TrackWrapper that is used for get is_child not by store """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._attributes = None

    @classmethod
    def from_pb_file(tkr, fpath):
        tkr = TrackWrapperWithoutStore()
        tracks = track3d_pb2.Tracks()
        with open(fpath, 'rb') as f:
            tracks.ParseFromString(f.read())
        # Should only be a single track in the pb file
        track = tracks.tracks[0]
        tkr._pid = str(track.pid_str)
        tkr.ts_vec = [f.timestamp for f in track.frames]
        tkr.pts_vec = \
            [(f.nose_pixels.x, f.nose_pixels.y) for f in track.frames]
        tkr.sit_vec = [f.detection_posture_type for f in track.frames]
        tkr.head_angles = [f.head_angle if not np.isnan(f.head_angle) else 0 for f in track.frames]
        tkr.height = [f.height_estimate for f in track.frames]
        tkr._cached_scores = {} # cached scores from various steps
        tkr.sort()
        tkr.get_movement()
        return tkr

    @property
    def attributes(self, ):
        return self._attributes

    @attributes.setter
    def attributes(self, pid_attribute):
        self._attributes = pid_attribute
    
    @property
    def is_child(self):
        CHILD_LS = [1, 2, 3]
        standing_height = [
            height
            for posture, height in zip(self.sit_vec, self.height)
            if posture != track3d_pb2.DetectionPostureType.DETECTION_POSTURE_SEATED and np.abs(height - 1.6) > 1e-3
        ]
        if not standing_height:
            return False
        total = 0.0
        for ls in CHILD_LS:
            if ls in self.attributes['life_stage']:
                total += self.attributes['life_stage'][ls]
        p = max(95, int(float(len(standing_height) - 25) / len(standing_height) * 100))
        p95_height = np.percentile(standing_height, p)
        if (p95_height < 1.6 and total >= 0.5) or total >= 0.9:
            ischild = True
        else:
            ischild = False
        return ischild

