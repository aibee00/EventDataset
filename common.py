import os

from SharedUtils import TrackWrapperWithoutStore

# 创建log管理器
import logging
logger = logging.getLogger(__name__)


def hms2sec(hms):
    h = int(hms / 10000)
    m = int(hms / 100) % 100
    s = hms % 100
    return h * 3600 + m * 60 + s


def ts_to_string(ts, sec_size=1, sep=":"):
    # from 40ms tic to XXhYYmZZs
    h = int(float(ts) / (sec_size * 60 * 60))
    m = int(float(ts) / (sec_size * 60)) % 60
    s = int(float(ts) / sec_size) % 60
    return "{0:02d}{3}{1:02d}{3}{2:02d}".format(h, m, s, sep)


# 定义duration, 如果开始时间大于结束时间，则返回0
def duration(time_range):
    if len(time_range) != 2:
        return 0
    
    start_time, end_time = time_range
    if start_time > end_time:
        return 0
    return end_time - start_time + 1


# 获取location
def get_location(pid_file):
    """ 读取proto，获取location
    """
    if not os.path.exists(pid_file):
        logger.error(f"{pid_file} not found!")
        return {}
    
    tkr = TrackWrapperWithoutStore.from_pb_file(pid_file)

    # only get locs
    pid_locs = {}
    pid2xy = {'loc': {}, 'vel': {}, 'acc': {}}
    for i in range(len(tkr.ts_vec)):
        pid2xy['loc'][int(tkr.ts_vec[i])] = tkr.pts_vec[i]
        pid2xy['vel'][int(tkr.ts_vec[i])] = tkr.vel_vec[i]
        pid2xy['acc'][int(tkr.ts_vec[i])] = tkr.acc_vec[i]
    pid_locs[tkr.pid] = pid2xy

    return pid_locs


def get_pid_locs(pid_tracks):
    """ Only parse locs of each pid from pid_tracks
    """
    pid_locs = {}
    for pid, tkr in pid_tracks.items():
        pid2xy = {'loc': {}, 'vel': {}, 'acc': {}}
        for i in range(len(tkr.ts_vec)):
            pid2xy['loc'][int(tkr.ts_vec[i])] = tkr.pts_vec[i]
            pid2xy['vel'][int(tkr.ts_vec[i])] = tkr.vel_vec[i]
            pid2xy['acc'][int(tkr.ts_vec[i])] = tkr.acc_vec[i]
        pid_locs[pid] = pid2xy
    return pid_locs

