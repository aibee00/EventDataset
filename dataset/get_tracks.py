#############################
# 加载并处理pid_output.tar
#############################
from glob import glob
import multiprocessing
import os
import os.path as osp
import sys
import logging
from SharedUtils import convert_unix_time_to_ts
from common import PoolSingleton, string_to_ts, ts_to_string

from proto import person_tracking_result_pb2
import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)


def read_proto(fname):
    """
    Read proto and return tracks of pid
    """
    tracks_from_pb = person_tracking_result_pb2.Track()
    with open(fname, 'rb') as f:
        tracks_from_pb.ParseFromString(f.read())
    return tracks_from_pb.pid, tracks_from_pb.single_view_tracks


def load_pid_tracks(pid_output, n_process=4):
    """ Loading tracks from pid_output/*.pb or tpid_output/*.pb
    Return:
        - pid2trk
    """
    # Read in per PID tracks
    logger.info("Reading per TPID tracks")
    tpid_files = sorted(glob(osp.join(pid_output, '*.pb')))
    logger.info("Total: {} pb in {}".format(len(tpid_files), pid_output))

    # single process or load pids under multi-mode
    if n_process == 1:
        locs = [read_proto(p) for p in tpid_files]
    else:
        pool = multiprocessing.Pool(n_process)
        pool = PoolSingleton(n_process).pool
        locs = pool.map(read_proto, tpid_files)
        pool.close()
        pool.join()
    
    # parse into pid-track map
    pid_locs = {}  # get the origin loc infos for interactions
    for pid, trks in locs:
        if trks is None:
            continue
        pid_locs.update({pid: trks})

    return pid_locs


def wrapper_xywh2xyxy(func):
    """
    Decorator for converting xywh to xyxy
    目的是用于保证is_xyxy的属性与对象的格式保持一致
    """
    def wrapped(self, *args, **kwargs):
        if self.is_xyxy:
            return self
        else:
            self.is_xyxy = True
            return func(self, *args, **kwargs)
    return wrapped


# 定义box对象
class BoundingBox:
    def __init__(self, x, y, w, h, s, t, ch, tid=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        
        self.score = s
        self.coords = [self.x, self.y, self.w, self.h]

        self.det_time_msec = t  # unix time
        self.ts = convert_unix_time_to_ts(t)  # chinese time
        self.bj = ts_to_string(self.ts)  # bj time
        self.channel = ch  # channel
        self.tid = tid  # track i

        self.__is_xyxy = False

    @property
    def is_xyxy(self):
        return self.__is_xyxy
    
    @is_xyxy.setter
    def is_xyxy(self, value):
        self.__is_xyxy = value
    
    @property
    def xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def xywh(self):
        return [self.x, self.y, self.w, self.h]

    @wrapper_xywh2xyxy
    def xywh2xyxy(self, ):
        """
        Convert (x, y, w, h) to (x1, y1, x2, y2)
        
        Return self
        """
        self.x1 = self.x
        self.y1 = self.y
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h
        self.coords = [self.x1, self.y1, self.x2, self.y2]
        return self
    
    def merge_by_avg(self, box):
        """
        Merge two BoundingBox objects
        """
        self.x = (self.x + box.x) / 2
        self.y = (self.y + box.y) / 2
        self.w = (self.w + box.w) / 2
        self.h = (self.h + box.h) / 2
        self.x1 = (self.x1 + box.x1) / 2
        self.y1 = (self.y1 + box.y1) / 2
        self.x2 = (self.x2 + box.x2) / 2
        self.y2 = (self.y2 + box.y2) / 2
        self.coords = [self.x1, self.y1, self.x2, self.y2] \
            if self.__is_xyxy else [self.x, self.y, self.w, self.h]
        return self
    
    def get_coords_as_tensor(self,):
        """
        Convert BoundingBox object to tensor
        """
        return torch.tensor(self.coords)
    
    def convert_coords_to_4_corners_points(self, ):
        """
        Convert BoundingBox object to 4-corners points coords
        
        Return: shape(4, 2)
        
        """
        self.coords = np.array([[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]])
        return self

    def __str__(self):
        return "({}, {}, {}, {}, {})".format(self.x, self.y, self.w, self.h, self.t)

    def __repr__(self):
        return "({}, {}, {}, {}, {})".format(self.x, self.y, self.w, self.h, self.t)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class TrackLoader:

    def __init__(self, pid_output_path):
        self.pid_output_path = pid_output_path
        self.tracks = {}  # pid2trk

    def load(self, path):
        """
        Load tracks from pid_output/*.pb or tpid_output/*.pb
        """
        logger.info("Loading tracks from {}".format(path))
        return load_pid_tracks(path)
    
    def load_single_pid_tracks(self, path, pid):
        """ Load single pid tracks from pid_output """
        proto_path = osp.join(path, '{}.pb'.format(pid))
        _pid, tkr = read_proto(proto_path)
        return tkr

    def get_svids(self, pid):
        """ Mapping pid to svids, get svids of pid at ts

        Return:
            - svids: list of svids
        
        Example:
            svids = [svid1, svid2, svid3, svid4]
        
        Note:
            - svids is not unique
            - svids is sorted by ts
        """
        # get all svids
        svids, _ = self.parse_pid_track_infos(pid)
        return svids

    def get_bboxes(self, pid, ts):
        """
        Get bounding boxes of pid at ts

        Return:
            - bboxes: list of boxes
        
        Example:
            bboxes = [box1, box2]
        
        Note:
            - box: class, box coords format: [x1, y1, x2, y2]
        """
        # get all bboxes
        _, bboxes = self.parse_pid_track_infos(pid)

        return bboxes.get(ts, [])

    def parse_pid_track_infos(self, pid):
        """
        Parse pid track infos
        
        Return:
            - svids: list of svids
            - bboxes: dict of bboxes, key is ts, value is list of box
        
        Example:
            svids = [svid1, svid2, svid3, svid4]
            bboxes = {
                ts1: [box1, box2],
                ts2: [box3, box4],
            }
        """
        # get track of pid
        pid_tracks = self.tracks.get(pid, None)
        if pid_tracks is None:
            pid_tracks = self.load_single_pid_tracks(self.pid_output_path, pid)
            # 更新到self.tracks中
            self.tracks.update({pid: pid_tracks})

        # get svids from track
        svids = set()

        # get bboxes from track
        bboxes = {}

        for trk in pid_tracks:
            for det in trk.detections:
                tid = trk.track_id  # tid示例: 'gacne_guangzhou_xhthwk-ch01001-fid-track-35356673-3c125728-33ee-4df3-9aad-f31626559146'
                channel = tid.split('-')[1]
                
                det_ts = int(convert_unix_time_to_ts(det.det_time_msec))

                # update svids
                svids.add(tid)

                # update bboxes
                box = BoundingBox(det.human_box_x,
                                  det.human_box_y,
                                  det.human_box_width,
                                  det.human_box_height,
                                  det.human_detection_score,
                                  det.det_time_msec,
                                  channel,
                                  tid).xywh2xyxy()
                bboxes.setdefault(det_ts, []).append(box)
        
        # sort bboxes by box.det_time_msec
        bboxes = {k: sorted(v, key=lambda x: x.det_time_msec) for k, v in bboxes.items()}

        return svids, bboxes
        


if __name__ == '__main__':
    pid_output_path = sys.argv[1]
    pid = sys.argv[2]
    # ts = sys.argv[3]
    # ts = string_to_ts(ts)
    loader = TrackLoader(pid_output_path)
    svids, bboxes = loader.parse_pid_track_infos(pid)
    print(svids)
    dets = bboxes[79181]
    for i, det in enumerate(dets):
        print(i, det)
        print()

