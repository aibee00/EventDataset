import datetime
import time
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from SharedUtils import TrackWrapperWithoutStore, convert_unix_time_to_ts
from proto import person_tracking_result_pb2

# 创建log管理器
import logging
import multiprocessing
import requests

from dataset.gen_grid_cameras_map import get_best_camera_views
import sys
logger = logging.getLogger(__name__)


STORE_INOUT_NEAR_TIME = 3  # 用于获取进出店时刻前后时间段[ts-NEAR_TIME, ts+NEAR_TIME]

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


def string_to_ts(string, sep=":"):
    if isinstance(string, datetime.time):
        return (string.hour * 3600) + (string.minute * 60) + string.second
    elif sep and sep in string:
        h,m,s = string.split(sep)
    else:
        h = string[0:2]
        m = string[3:5]
        s = string[6:8]
    return int(h)*3600 + int(m)*60 + int(s)


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


# get location of pid_output pb, convert to location format same as new_xy_locs
def get_location_pid_output(fname):
    """
    Read proto and return tracks of pid
    """
    tracks_from_pb = person_tracking_result_pb2.Track()
    with open(fname, 'rb') as f:
        tracks_from_pb.ParseFromString(f.read())

    # only get locs
    pid_locs = {}
    pid2xy = {'loc': {}, 'vel': {}, 'acc': {}}
    locs = {}
    for trk in tracks_from_pb.single_view_tracks:
        for det in trk.detections:
            ts = int(convert_unix_time_to_ts(det.det_time_msec))
            locs.setdefault(ts, []).append(np.array([det.map_pos.x, det.map_pos.y]))

    # get avg loc for each ts
    for ts in locs:
        locs[ts] = np.mean(locs[ts], axis=0)

    locs = sorted(locs.items(), key=lambda x: x[0])

    for ts, loc in locs:
        pid2xy['loc'][ts] = loc
        pid2xy['vel'][ts] = 0.0
        pid2xy['acc'][ts] = 0.0
        pid_locs[tracks_from_pb.pid] = pid2xy

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


def get_overlap_time(time_range1, time_range2):
    """ 返回time_range1和time_range2的overlap时间
    """
    start_time, end_time = time_range1
    start_time2, end_time2 = time_range2
    if start_time > end_time or start_time2 > end_time2:
        return [0, 0]
    if end_time < start_time2 or end_time2 < start_time:
        return [0, 0]
    return max(start_time, start_time2), min(end_time, end_time2)


def get_pid_loc_time_range(pid_locs, pid):
    """ 返回pid_loc的start_time到end_time

    Args:
        pid_locs (dict): pid_loc
        pid (str): pid

    Returns:
        tuple: (start_time, end_time)
    
    """
    if pid not in pid_locs:
        return 0, 0
    
    pid_xy = pid_locs[pid]
    ts_vec = sorted(pid_xy['loc'].keys())
    if len(ts_vec) == 0:
        return 0, 0
    start_time = ts_vec[0]
    end_time = ts_vec[-1]

    return start_time, end_time

# 定义一个自动换行的函数，最大长度为max_length, 并将多行文本保存到一个列表中返回
def auto_wrap(text, max_length, textSize, sep=";"):
    lines = []
    line = ''
    lineLen = 0

    bar = text if sep != ";" else text.split(sep)
    for word in bar:
        # 判断word是不是字母
        wlen = len(word) // 2 if word.isalpha() else len(word)
        
        # 更新lineLen
        lineLen += wlen
        
        if lineLen * textSize > max_length - wlen * textSize:
            lines.append(line)
            line = ''
            lineLen = 0

        line += word + sep
    lines.append(line)
    return lines
    

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(osp.join(osp.dirname(__file__), "simsun.ttc"), textSize, encoding="utf-8")

    # 将文本prompt画到img_blank上, 并调根据剩余空间调整字体大小使之填满整个img_blank区域，自动换行
    # 当字体超过img的宽度时自动换行
    events = auto_wrap(text, img.size[0], textSize)
    
    for event in events:
        lines = auto_wrap(event, img.size[0], textSize, sep="")
        for line in lines:
            print(f"line: {line}")

            # 如果超出了img的高度，则img在底部concat一块高度为(textSize + 8)、宽度为img.size[0]的空白区域
            if position[1] + textSize + 8 > img.size[1]:
                img_blank = Image.new('RGB', (img.size[0], (textSize + 8)), (255, 255, 255))
                img = np.concatenate((img, img_blank), axis=0)
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
            
            draw.text(position, line, textColor, font=fontStyle)
            position = (position[0], position[1] + textSize + 8)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



# 定义一个python3实现单例模式的类
class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

# 定义一个multiprossing.Pool的单例模式的类
class PoolSingleton(Singleton):
    def __init__(self, processes=4):
        self.pool = multiprocessing.Pool(processes=processes)
    

# 对bboxes内所有bbox各个坐标求平均，合并为一个bbox返回
def merge_bboxes_at_one_channel(bboxes, channel=None, score_thres=0.5):
    """
    bboxes: [bbox1, bbox2, ...]
    bbox: class BoundingBox

    Return a dict of merged bboxes.
    key: channel, value: BoundingBox
    
    Example:
    >>> bboxes = [BoundingBox(1, 1, 1, 1, 1, 1), BoundingBox(2, 2, 2, 2, 2, 2)]
    >>> merge_bboxes_at_each_channel(bboxes)
    {1: BoundingBox(1, 1, 1, 1, 1, 1), 2: BoundingBox(2, 2, 2, 2, 2, 2)}
    
    >>> bboxes = [BoundingBox(1, 1, 1, 1, 1, 1), BoundingBox(2, 2, 2, 2, 2, 2), BoundingBox(3, 3, 3, 3, 3, 3)]
    >>> merge_bboxes_at_each_channel(bboxes)
    {1: BoundingBox(1, 1, 1, 1, 1, 1), 2: BoundingBox(2, 2, 2, 2, 2, 2), 3: BoundingBox(3, 3, 3, 3, 3, 3)}
    """
    # 使用numpy实现求所有bbox的各个坐标的平均值
    ch_bbox_map = {}  # k: channel, v: [BoundingBox...]
    bbox_merge = {}  # k: channel, v: BoundingBox

    # fill ch_bbox_map with data
    for bbox in bboxes:
        ch_bbox_map.setdefault(bbox.channel, []).append(bbox)
    
    # merge at each channel
    for ch, _bboxes in ch_bbox_map.items():
        if channel and ch != channel:
            continue
        
        # 按照bbox.score由高到低排序
        _bboxes = sorted(_bboxes, key=lambda bbox: bbox.score, reverse=True)

        box_m = _bboxes[0]
        for bbox in _bboxes[1:]:
            if bbox.score < score_thres:
                continue

            box_m.merge_by_avg(bbox)
        
        # 更新bbox_merge
        bbox_merge[ch] = box_m

    return bbox_merge


# 给出pid、location和时间段，返回覆盖到该location的所有channels
def get_cover_channels(grid_cameras_map, pid, loc, time_slice):
    """ 获取在时间段time_slice内pid所在位置有覆盖的cameras

    Args:
        grid_cameras_map: 预先生成的grid to cameras的映射关系
        pid: 某个人的pid
        loc: 轨迹文件
        time_slice: 可以是一段时间[st, et]或一个时刻ts
    Returns:
        channels: set of channels
    """
    channels = dict()
    if isinstance(time_slice, list):
        for ts in range(time_slice[0], time_slice[1]):
            if pid not in loc:
                continue
            if ts not in loc[pid]['loc']:
                continue

            location = loc[pid]['loc'][ts]
            best_cameras = get_best_camera_views(location, grid_cameras_map)
            for ch in best_cameras:
                channels.setdefault(ch, 0)
                channels[ch] += 1
        channels = {ch: channels[ch] for ch in channels if channels[ch] > 0}
        channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)
        channels = [ch for ch, _ in channels]
    else:
        if pid not in loc:
            return channels
        if time_slice not in loc[pid]['loc']:
            return channels

        location = loc[pid]['loc'][time_slice]
        best_cameras = get_best_camera_views(location, grid_cameras_map)
        return best_cameras

    return channels


# 根据开始时间获取时间段
def get_video_time_range(start_time, offset):
    """
    start_time: 开始时间
    offset: 如[10, 20]
    :return: tuple(int, int)
    """
    if isinstance(start_time, str):
        start_time = hms2sec(int(start_time))
    end_time = start_time + offset
    return start_time, end_time


# 定义get_overlap_time_range函数，输入两个时间段，返回overlap时间段
def get_overlap_time_range(time_range1, time_range2):
    """
    time_range1: (start_time1, end_time1)
    time_range2: (start_time2, end_time2)
    """
    if time_range1[0] >= time_range2[1] or time_range1[1] <= time_range2[0]:
        return None
    start_time = max(time_range1[0], time_range2[0])
    end_time = min(time_range1[1], time_range2[1])
    return start_time, end_time


# 计算距离给定时间最近的整15分钟点
def get_nearest_slice_start_time(given_time, interval=15):
    """
    time: 时间戳
    """
    if isinstance(given_time, int):
        given_time = ts_to_string(given_time)
    # 给定的时间
    if isinstance(given_time, str):
        given_time = datetime.datetime.strptime(given_time, "%H:%M:%S")
    elif isinstance(given_time, datetime.time):
        given_time = datetime.datetime.combine(datetime.date.today(), given_time)
    
    # 计算距离给定时间最近的整15分钟点
    nearest_15_min = given_time - datetime.timedelta(minutes=given_time.minute % interval, seconds=given_time.second)
    # # 转换为字符串表示
    # nearest_15_min_str = nearest_15_min.strftime("%H:%M:%S")
    return nearest_15_min.time()


# 为某个对象增加(或者更新/设置)多个新的属性
def set_properties(obj, **properties):
    if isinstance(obj, dict):
        obj.update(properties)
    else:
        for key, value in properties.items():
            setattr(obj, key, value)
    return obj


# 下载文件包
def download(url, filename):
    """
    下载文件

    Args:
        url: 文件url
        filename: 文件路径
    """
    filename = Path(filename)
    if not filename.exists():
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {url} to {filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                chunk_size = 8192  # 增大 chunk_size
                downloaded_size = 0
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            # 更新进度条
                            done = int(50 * downloaded_size / total_size)
                            sys.stdout.write('\r[{}{}] {:.2f}%'.format('█' * done, '.' * (50 - done), (downloaded_size / total_size) * 100))
                            sys.stdout.flush()
                    sys.stdout.write('\n')
            print("Download completed.")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP错误: {err}")
            raise err
        except requests.exceptions.RequestException as err:
            print(f"下载过程中出错: {err}")
            raise err
        except Exception as err:
            print(f"发生错误: {err}")
            raise err
        

def is_directory_empty(directory):
    """检查指定的目录是否为空"""
    # 使用 pathlib.Path.iterdir() 迭代目录中的内容
    return not any(Path(directory).iterdir())