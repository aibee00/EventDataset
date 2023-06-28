import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from SharedUtils import TrackWrapperWithoutStore

# 创建log管理器
import logging
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
