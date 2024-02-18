import os
import os.path as osp
from pathlib import Path
import json
import argparse
from typing import List
import vipy, vipy.dataset


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--cap_dataset_path', default='/training/wphu/Dataset/Cap/cap_classification_clip', type=str, help='path of cap dataset')
    return parser.parse_args()


class CapDataset:
    def __init__(self, root):
        self.root = root
        self.video_dir = osp.join(root, 'videos')
        self.label_dir = osp.join(root, 'annotations')
        self.video_list = os.listdir(self.video_dir)
        self.label_list = os.listdir(self.label_dir)
        self.video_list.sort()
        self.label_list.sort()

        self.videos = self.load_videos()  # load all videos object

    
    def load_videos(self, file_name):
        videos = {}
        for file_name in self.label_list:
            V = vipy.util.load(osp.join(self.label_dir, file_name))  # video list, 一个label对应几千个video_clips, label例如 person_enters_car
            videos[file_name] = V
        return videos
    
    def get_video_info(self, V: List[vipy.video.Scene]):
        d = vipy.dataset.Dataset(V)
        h = d.histogram()  # class frequencies
        f = [v.framerate() for v in V]  # video framerates 
        T = [[(bb, bb.category(), bb.xywh()) for t in v.tracklist() for bb in t] for v in V]  # all bounding boxes in (xmin, ymin, width, height) format at video framerate (this will take a while)
        A = [[y for y in v.annotation()] for v in V]  # all framewise annotation (this will take a while)
        return d, h, f, T, A

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = osp.join(self.video_dir, self.video_list[index])
        label_path = osp.join(self.label_dir, self.label_list[index])
        with open(label_path, 'r') as f:
            V = vipy.util.load(f)
        return video_path, V


