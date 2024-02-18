""" 格式转换，from cap to lavis
Data path tree:
/training/wphu/Dataset/Cap/cap_classification_clip
    - annotaions
        - activity_1.json  # label and meta info of video
        - activity_2.json
    - videos
        - activity_1
            - video_1.mp4
            - video_2.mp4
            - ...
        - activity_2
            - video_3.mp4
            - video_4.mp4
            - ...
        - ...
    - train_val.json: {'train': ['video_1', 'video_2', ...], 'val': ['video_3', 'video_4', ...]}
    - cap_activity_classification_ref.csv: Each line: video_file_id, frame_rate, activity_id

To introspect videos:

```python
import vipy, vipy.dataset

V = vipy.util.load('/path/to/annotations.json')
h = vipy.dataset.Dataset(V).histogram()  # class frequencies
f = [v.framerate() for v in V]  # video framerates 
T = [[(bb, bb.category(), bb.xywh()) for t in v.tracklist() for bb in t] for v in V]  # all bounding boxes in (xmin, ymin, width, height) format at video framerate (this will take a while)
A = [[y for y in v.annotation()] for v in V]  # all framewise annotation (this will take a while)
```

lavis dataset format:
List[Dict[
    "image": str,
    "image_id": str,
    "bbox": List[float],
    "dense_caption": {caption}<bbox>{bbox},
]]
"""
import os
import json
import argparse
import logging

from pathlib import Path

import vipy
from vipy.activity import Activity


class Cap2Lavis(object):
    def __init__(self, cap_dir, lavis_dir):
        self.cap_dir = cap_dir
        self.annotation_dir = os.path.join(cap_dir, "annotations")
        self.video_dir = os.path.join(cap_dir, "videos")
        self.train_val = json.loads(open(os.path.join(cap_dir, 'train_val.json'), 'r').read())
        self.lavis_dir = lavis_dir

        self.sample_num_per_clip = 1  # 每个clip采样几张图片, 共145万个clips
        self.sample_num_per_activity = 500  # 每个动作采样几个clips

        self.valid_activity_names = self._load_valid_activity_names()

    @property
    def train_video_names(self,):
        return self.train_val['train']
    
    @property
    def val_video_names(self,):
        return self.train_val['val']
    
    @property
    def activity_names(self, ):
        """
        Return:
            List[str]: activity_names, e.g. ['activity_1', 'activity_2', ...]
        """
        return [fp.stem for fp in Path(self.annotation_dir).glob('*.json')]
    
    def _load_valid_activity_names(self,):
        with open(Path(__file__).parent / 'valid_activities.txt', 'r') as f:
            lines = f.readlines()
            valid_activity_names = [line.strip() for line in lines]
        return valid_activity_names
    
    def get_name_from_path(self, video_path_name):
        return video_path_name.split('/')[-1].split('.')[0]  # video_1.mp4 -> video_1, video_2.mp4 -> video_2, ...
    
    def load_activity(self, activity_name):
        """
        Return:
            V: List[vipy.video.Scene[vipy.image.Scene[vipy.image.ImageDetection]]]
        """
        # load activity from activity_name
        activity_path = os.path.join(self.annotation_dir, activity_name + '.json')
        V = vipy.util.load(activity_path)
        return V  # <class 'vipy.activity.Activity'>, 一个 activity中包含多个 video.Scene(clip)
    
    def process_trainset(self,):
        pass

    def process_valset(self,):
        pass
    
    def convert(self):
        """ Convert cap to lavis format.

        如何用cap数据生成基于图片的lavis格式的数据集?
        1. 所有的动作都要有
        2. 每个动作需要采样出一定数量的图片作为训练数据,一个动作有几千个clip,
            假设一个clip采样4张(因为每个clip基本不超过4秒, 每秒采样一张)图则每个动作就会有几千甚至上万张图片.
        """
        for activity_name in self.valid_activity_names:
            V = self.load_activity(activity_name)
            for v in V:
                if self.get_name_from_path(v.filename()) not in self.train_video_names:
                    # train set
                    self.process_trainset(v, activity_name)
                else:
                    self.process_valset(v, activity_name)
        






