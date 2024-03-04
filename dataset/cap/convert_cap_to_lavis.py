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
import multiprocessing

from tqdm import tqdm
from pathlib import Path

import vipy
from vipy.activity import Activity


def worker_helper(args):
    obj, activity_name, video_info = args
    return obj._worker_process(activity_name, video_info)


class ConvertCapToLavis(object):
    def __init__(self, cap_dir, lavis_dir, max_clip_num=500, num_procs=16, valid_activity_path=None):
        self.cap_dir = cap_dir
        self.annotation_dir = os.path.join(cap_dir, "annotations")
        self.video_dir = os.path.join(cap_dir, "videos")
        self.train_val = json.loads(open(os.path.join(cap_dir, 'train_val.json'), 'r').read())

        self.lavis_dir = Path(lavis_dir)
        self.lavis_image_dir = self.lavis_dir / "images"
        self.lavis_image_dir.mkdir(parents=True, exist_ok=True)
        self.lavis_anno_dir = self.lavis_dir / "annotations"
        self.lavis_anno_dir.mkdir(parents=True, exist_ok=True)

        self.train_labels = []
        self.val_labels = []

        self.sample_num_per_clip = 4  # 每个clip采样几张图片, 共145万个clips
        self.max_clip_num = max_clip_num  # 每个动作采样几个clips

        self.valid_activity_names = get_activities_not_processed(
            cap_dir, lavis_dir
        ) #self._load_valid_activity_names(valid_activity_path)

        self.num_procs = num_procs

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
    
    def _load_valid_activity_names(self, valid_activity_path):
        if valid_activity_path is not None and os.path.exists(valid_activity_path):
            with open(valid_activity_path, 'r') as f:
                lines = f.readlines()
                valid_activity_names = [line.strip() for line in lines]
        else:
            valid_activity_names = os.listdir(self.video_dir)
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
    
    def _normlize_bbox(self, bbox, H, W, decimal=3):
        """
        Args:
            bbox: vipy.geometry.BoundingBox

        Return:
            List[float]: normalized bbox, e.g. [0.0, 0.0, 0.0, 0.0]
        """
        bbox = [
            round(bbox.xmin() / W, decimal), 
            round(bbox.ymin() / H, decimal), 
            round(bbox.xmax() / W, decimal), 
            round(bbox.ymax() / H, decimal)]
        return bbox
    
    def get_bbox(self, frame):
        """
        Args:
            frame: vipy.image.Scene

        Return:
            List[float]: bbox, e.g. [0.0, 0.0, 0.0, 0.0]
        """
        bbox = frame.boundingbox()
        if not bbox:
            return bbox
        H, W = frame.height(), frame.width()
        bbox = self._normlize_bbox(bbox, H, W)
        return bbox
    
    def _process(self, v, activity_name, istrain=True):
        """
        Args:
            v: vipy.video.Scene[vipy.image.Scene[vipy.image.ImageDetection]]
            activity_name: str

        How to process?
            1. 对每个clip抽取self.sample_num_per_clip 数量的图片
            2. 获取到每张图片的 bbox 坐标
            3. 将 activity_name 作为每张图片的 caption.
            4. 生成lavis数据集格式的label
        """
        # process trainset
        video_name = self.get_name_from_path(v.filename())

        # 对每个clip抽取self.sample_num_per_clip 数量的图片
        num_frames = len(v.framelist())  # 一个clip中包含多个bbox
        interval = num_frames // self.sample_num_per_clip  # 每个clip采样几张图片
        for i in range(0, num_frames, interval):
            # 获取到每张图片的 bbox 坐标
            cur_frame = v.frame(i)  # vipy.image.Scene
            caption = cur_frame.category()
            bbox = self.get_bbox(cur_frame)  # List[float]
            if not bbox:
                continue  # skip empty bbox. e.g. 'activity_1<bbox>[0.0, 0.0, 0.0, 0.0]'

            # 将 activity_name 作为每张图片的 caption.
            caption = f'{activity_name}<bbox>{bbox}'

            # 保存
            image_path = self.lavis_image_dir / activity_name
            image_path.mkdir(parents=True, exist_ok=True)
            image_name = f'{video_name}_frame_{i}.jpg'
            image_dir = image_path / image_name
            if not image_dir.exists():
                img = vipy.image.Image(array=cur_frame.numpy(), colorspace='rgb')
                img.savefig(image_dir)

            # update annotation to train set
            image_id = len(self.train_labels) if istrain else len(self.val_labels)
            annotation = {
                "image": f"{activity_name}/{image_name}",
                "image_id": image_id,
                "bbox": bbox,
                "dense_caption": caption,  # e.g. 'activity_1<bbox>[0.0, 0.0, 0.0, 0.0]'
            }
            if istrain:
                self.train_labels.append(annotation)
            else:
                self.val_labels.append(annotation)

    def _worker_process(self, activity_name, video_info):
        """ Process a single video clip in a worker process.
        
        Args:
            v: vipy.video.Scene[vipy.image.Scene[vipy.image.ImageDetection]]
            activity_name: str

        How to process?
            1. 对每个clip抽取self.sample_num_per_clip 数量的图片
            2. 获取到每张图片的 bbox 坐标
            3. 将 activity_name 作为每张图片的 caption.
            4. 生成lavis数据集格式的label
        """
        v, is_train = video_info
        annotations = []

        # process trainset
        video_name = self.get_name_from_path(v.filename())

        # 对每个clip抽取self.sample_num_per_clip 数量的图片
        num_frames = len(v.framelist())  # 一个clip中包含多个bbox
        interval = num_frames // self.sample_num_per_clip  # 每个clip采样几张图片
        for i in range(0, num_frames, interval):
            # 获取到每张图片的 bbox 坐标
            cur_frame = v.frame(i)  # vipy.image.Scene
            caption = cur_frame.category()
            bbox = self.get_bbox(cur_frame)  # List[float]
            if not bbox:
                continue  # skip empty bbox. e.g. 'activity_1<bbox>[0.0, 0.0, 0.0, 0.0]'

            # 将 activity_name 作为每张图片的 caption.
            caption = f'{activity_name}<bbox>{bbox}'

            # 保存
            image_path = self.lavis_image_dir / activity_name
            image_path.mkdir(parents=True, exist_ok=True)
            image_name = f'{video_name}_frame_{i}.jpg'
            image_dir = image_path / image_name
            if not image_dir.exists():
                img = vipy.image.Image(array=cur_frame.numpy(), colorspace='rgb')
                img.savefig(image_dir)

            # update annotation to train set
            image_id = len(self.train_labels) if is_train else len(self.val_labels)
            annotation = {
                "image": f"{activity_name}/{image_name}",
                "image_id": image_id,
                "bbox": bbox,
                "dense_caption": caption,  # e.g. 'activity_1<bbox>[0.0, 0.0, 0.0, 0.0]'
            }
            annotations.append(annotation)
        return (is_train, annotations)

    def update_label_file(self, istrain=True):
        """ Save labels """
        if istrain:
            label_path = self.lavis_anno_dir / 'label_train_cap2lavis.json'
        else:
            label_path = self.lavis_anno_dir / 'label_val_cap2lavis.json'

        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_labels if istrain else self.val_labels, f, indent=4, ensure_ascii=False)
        print(f"Save labels to {label_path}")
        
    
    def convert(self):
        """ Convert cap to lavis format.

        如何用cap数据生成基于图片的lavis格式的数据集?
        1. 所有的动作都要有
        2. 每个动作需要采样出一定数量的图片作为训练数据,一个动作有几千个clip,
            假设一个clip采样4张(因为每个clip基本不超过4秒, 每秒采样一张)图则每个动作就会有几千甚至上万张图片.
        """
        for activity_name in tqdm(self.valid_activity_names, desc=f"Activities"):
            print(f"Processing activity: {activity_name}")
            V = self.load_activity(activity_name)
            
            processed_clips_base_name = set()  # 这里因为每条clip有3个重复的video场景差不多拍摄了3条，我们只需要取其中一条即可
            
            for v in tqdm(V[:self.max_clip_num], desc="Clips"):
                base_name = self.get_name_from_path(v.filename()).split('_')[0]
                if base_name in processed_clips_base_name:
                    continue

                processed_clips_base_name.add(base_name)

                if self.get_name_from_path(v.filename()) not in self.train_video_names:  # train set
                    # print(f"Processing train set, video: {v.filename()}")
                    self._process(v, activity_name, istrain=True)  # process trainset. e.g. [{}, {}, {}, {}]
                    # update label file
                    self.update_label_file(istrain=True)
                else:
                    # print(f"Processing val set, video: {v.filename()}")
                    self._process(v, activity_name, istrain=False)
                    # update label file
                    self.update_label_file(istrain=False)
    
    def convert_multiprocess(self):
        """ Convert cap to lavis format.

        如何用cap数据生成基于图片的lavis格式的数据集?
        1. 所有的动作都要有
        2. 每个动作需要采样出一定数量的图片作为训练数据,一个动作有几千个clip,
            假设一个clip采样4张(因为每个clip基本不超过4秒, 每秒采样一张)图则每个动作就会有几千甚至上万张图片.
        """
        pool = multiprocessing.Pool(self.num_procs)

        for activity_name in tqdm(self.valid_activity_names, desc=f"Activities"):
            print(f"Processing activity: {activity_name}")
            V = self.load_activity(activity_name)
            
            processed_clips_base_name = set()  # 这里因为每条clip有3个重复的video场景差不多拍摄了3条，我们只需要取其中一条即可
            tasks = []
            
            for v in tqdm(V[:self.max_clip_num], desc="Clips"):
                base_name = self.get_name_from_path(v.filename()).split('_')[0]
                if base_name in processed_clips_base_name:
                    continue

                processed_clips_base_name.add(base_name)

                is_train = self.get_name_from_path(v.filename()) in self.train_video_names
                tasks.append((self, activity_name, (v, is_train)))

            # Use multiprocessing pool
            results = pool.map(worker_helper, tasks)

            # Collect results and update shared resources
            for is_train, annotations in results:
                if is_train:
                    self.train_labels.extend(annotations)
                else:
                    self.val_labels.extend(annotations)
            
            # save result
            self.update_label_file(istrain=True)
            self.update_label_file(istrain=False)

        pool.close()
        pool.join()

    def run(self,):
        if self.num_procs > 1:
            self.convert_multiprocess()
        else:
            self.convert()
        

def get_activities_not_processed(cap_dir, lavis_dir):
    """ Get activities not processed. """
    cap_activity_names = os.listdir(os.path.join(cap_dir, "videos"))
    lavis_activity_names = os.listdir(os.path.join(lavis_dir, "images"))
    return list(set(cap_activity_names) - set(lavis_activity_names))  # 返回cap中没有被处理的动作名称列表. e.g. ['activity_1', 'activity_2']


if __name__ == "__main__":
    cap_dir = '/training/wphu/Dataset/Cap/cap_classification_clip'
    lavis_dir = '/training/wphu/Dataset/lavis/from_cap/'
    valid_activity_path = None  # "/ssd/wphu/chatglm/EventDataset/dataset/cap/valid_activities.txt"
    engine = ConvertCapToLavis(
        cap_dir, 
        lavis_dir, 
        max_clip_num=100000,  # 每种动作采样多少个clips, 超过这个值时截断
        num_procs=32,  # 进程数量
        valid_activity_path=valid_activity_path,  # 设置为None时默认使用所有video
    )
    engine.run()
    print('Done')
    print(f'lavis_dir: {lavis_dir}')
    print(f'lavis_image_dir: {lavis_dir}/images')
    print(f'lavis_anno_dir: {lavis_dir}/annotations')



