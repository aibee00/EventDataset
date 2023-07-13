"""
Date: 20230630
Author: wphu
Description:
    This script is used to extract images from videos.
    The videos are stored in the local directory.
    The extracted images are stored in the local directory.
Note:
    The file {}.final.reduced.json must be contained in the video path. 
    It stores all pids and there bbox infos.
"""
import cv2
import os
from random import random
import glob
import os.path as osp
import json
from aibee_hdfs import hdfscli
import argparse
import numpy as np
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_infos_path', type=str, help="output path", default="./data_infos_labeling/")
    parser.add_argument('--username', type=str, required=False, default="wphu", help='job owner name')
    parser.add_argument('--num_samples', type=int, required=False, default=10000, help='number samples you want to gen')
    parser.add_argument('--diplacement_consistency_window', type=int, help='diplacement_consistency_window', default=5)
    return parser.parse_args()

args = parse_arguments()
assert args.username is not None, \
    "When running in non local mode, username is required"
keytab = "/home/{}/{}.keytab".format(args.username, args.username)
hdfscli.initKerberos(keytab, args.username)
client = hdfscli.HdfsClient(user=args.username)

"""
 Read video and convert it into pictures
 读取视频并将视频转换为图片
 注意请勿在视频文件夹中放其他文件
"""

def ts_to_string(ts, sec_size=1, sep=":"):
    # from 40ms tic to XXhYYmZZs
    h = int(float(ts) / (sec_size * 60 * 60))
    m = int(float(ts) / (sec_size * 60)) % 60
    s = int(float(ts) / sec_size) % 60
    return "{0:02d}{3}{1:02d}{3}{2:02d}".format(h, m, s, sep)


class ExtractImagesFromVideos(object):
    def __init__(self, args):
        self.args = args
        self.color_map = [[0,255,0], [255,0,0], [0,0,255], [0,180,0], [180,0,0],[0,0,180], [0,0,140]]

    @staticmethod
    def hmsstr_to_ts(str):
        h = str[0:2]
        m = str[2:4]
        s = str[4:6]
        return int(h)*3600 + int(m)*60 + int(s)

    def save_img(self, video_path, data_infos_path, ts_to_extract):
        # video_path = r'E:/VideoDir/'
        # videos = os.listdir(video_path)
        # videos = load_video(video_path)
        # for video_name in videos[:1]:
            video_name = video_path
            video_name_dir = video_name.split('.')[0]
            channel_name = video_name_dir.split('/')[-1]
            ch_name = channel_name.split('_')[0]
            video_time = channel_name.split('_')[1]
            year_month_day = video_time[:8]
            start_time = video_time[8:]
            start_ts = self.hmsstr_to_ts(start_time)

            # print("Channel: {}".format(channel_name))
            vc = cv2.VideoCapture(video_name)  # 读入视频文件
            c = 1
            if vc.isOpened():  # 判断是否正常打开
                rval, frame = vc.read()
            else:
                rval = False
            timeF = 12  # 视频帧计数间隔频率
            while rval:  # 循环读取视频帧
                rval, frame = vc.read()
                if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                    cur_ts = start_ts + c // timeF
                    t_name = ''.join(ts_to_string(start_ts + c // timeF).split(":"))
                    # print(ts_to_string(cur_ts), cur_ts in ts_to_extract, "start: {}, end: {}".format(ts_to_string(ts_to_extract[0]), ts_to_string(ts_to_extract[-1])))
                    if frame is not None and cur_ts in ts_to_extract:
                        # print("Saving image at {}".format(ts_to_string(cur_ts)))
                        print(ts_to_string(cur_ts))
                        img_dir = osp.join(data_infos_path, ch_name + '_' + year_month_day + t_name + '.jpg')
                        
                        if not osp.exists(img_dir):
                            # Add body patches on img
                            reduced_json = "{}.final.reduced.json".format(video_path)
                            bboxes = json.loads(open(reduced_json, 'rb').read())
                            if c in bboxes or str(c) in bboxes:
                                cur_bboxes = bboxes[c] if c in bboxes else bboxes[str(c)]
                                for i,(bbox, pid) in enumerate(cur_bboxes):
                                    bbox = np.array(bbox, dtype=int)
                                    color = (0 * random(), 255 * random(),255 * random()) if i > 6 else self.color_map[i]
                                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 3)
                                    cv2.putText(frame, pid, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                            # Save img
                            cv2.imwrite(osp.join(data_infos_path, ch_name + '_' + year_month_day + t_name + '.jpg'), frame)
                        else:
                            print(" {} has existed!".format(img_dir))  
                c = c + 1
                # cv2.waitKey(1)
            vc.release()

    def load_video(self, video_path):
        input_path_extension = video_path.split('.')[-1]
        if input_path_extension in ['mp4']:
            return [video_path]
        elif input_path_extension == "txt":
            with open(video_path, "r") as f:
                return f.read().splitlines()
        else:
            return glob.glob(os.path.join(video_path, "*.mp4"))

    def extract_imgs(self, bmk):
        # Extract images from videos
        # video_images_path = osp.join(self.args.data_infos_path, 'video_images', bmk)  # extract ts img according to ts_list
        video_images_path = osp.join(self.args.data_infos_path, 'video_images_v2', bmk)  # extract ts img according to [ts-5,ts+5]
        if not os.path.exists(video_images_path):
            os.makedirs(video_images_path)

        data_infos = json.loads(open(os.path.join(self.args.data_infos_path, 'data_infos_{}.json'.format(bmk)), 'rb').read())

        ## First, get all ts_set of each pair, only extract time in ts_set
        video_tslist_map = {}
        for pid in data_infos:
            for staff in data_infos[pid]:
                ts_set = set()
                channels = set()
                for ts in data_infos[pid][staff]:
                    # cur_tslist = data_infos[pid][staff][ts]["consistent_window_ts"]
                    ts_set |= set(range(int(ts) - self.args.diplacement_consistency_window, int(ts) + self.args.diplacement_consistency_window))
                    video_path = data_infos[pid][staff][ts]["video_path"]
                    if not video_path:
                        continue
                    video_tslist_map.setdefault(pid, {})
                    video_tslist_map[pid].setdefault(staff, {})
                    video_tslist_map[pid][staff].setdefault(video_path, set())
                    video_tslist_map[pid][staff][video_path].add(int(ts))

        num = 0
        num_samples = self.args.num_samples
        for pid in tqdm(video_tslist_map):
            if num == num_samples:
                break
            for staff in video_tslist_map[pid]:
                if num == num_samples:
                    break
                for video_path in video_tslist_map[pid][staff]:
                    ts_to_extract = sorted(video_tslist_map[pid][staff][video_path])
                    if len(ts_to_extract) < 15:
                        continue
                    video_mp4 = video_path.split('/')[-1]
                    video_name = video_mp4.split('.')[0]
                    save_path = osp.join(video_images_path, video_name)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    
                    print("pid: {}, staff: {}, video_path: {}, ts_to_extract length: {}".format(
                        pid, staff, video_path, len(ts_to_extract)))
                    if not osp.exists(video_path):
                        hdfs_body_path = "/bj_dev/prod/QA/qa_store/store_benchmarks/{}/{}/{}v7/{}/processed/body/{}/{}".format(*(bmk.split('-')), bmk.split('-')[-1], video_mp4)
                        video_face = video_name[:-4] + "*" + ".mp4.cut.mp4"
                        hdfs_face_path = "/bj_dev/prod/QA/qa_store/store_benchmarks/{}/{}/{}v7/{}/processed/face/{}/{}".format(*(bmk.split('-')), bmk.split('-')[-1],video_face)
                        print("Warning: {} not found! Downloading from hdfs: {}".format(video_path, hdfs_body_path))
                        try:
                            client.download(hdfs_body_path, video_path)
                        except:
                            client.download(hdfs_face_path, video_path)

                    self.save_img(video_path, save_path, ts_to_extract)
                    num += 1
                    print(num)
                    if num == num_samples:
                        break

    def __call__(self, bmk):
        return self.extract_imgs(bmk)


if __name__ == '__main__':
    bmks_train = ["VW-changchun-rq-20210728"]
    bmks_test = ["GACNE-guangzhou-xhthwk-20210717"]

    if not os.path.exists(args.data_infos_path):
        os.makedirs(args.data_infos_path)

    extractor = ExtractImagesFromVideos(args)
    # for bmk in bmks_train + bmks_test:
    for bmk in bmks_train:
        extractor(bmk)