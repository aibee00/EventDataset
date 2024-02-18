"""
随机从图片列表中选取一定数量的图片，将其俄存到指定文件处
"""

import os
from pathlib import Path
import sys
from glob import glob
import json
import random
from typing import Any

VERSION="v2"

# label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_onlyperson_train.json"
label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_train.json"
label_path = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json"
save_path = f"/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list_{VERSION}.json"
save_path = f"/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/train_img_list_expand.json"


if len(sys.argv) > 1:
    label_path = sys.argv[1]
    save_path = sys.argv[2]


class DataInfoGen():

    def __init__(self, label_path, save_path, image_path=None) -> None:
        self.label_path = label_path
        self.save_path = save_path

        if Path(self.save_path).name.endswith('.json') and not Path(self.save_path).parent.exists():
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        elif Path(self.save_path).is_dir() and not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        if image_path is None:
            root_path = Path(label_path).parent.parent / 'images'
            if not root_path.exists():
                root_path = Path(label_path).parent / 'images'
                if not root_path.exists():
                    raise ValueError(f'root path {root_path} not exists')
            self.image_path = root_path
        else:
            self.image_path = image_path

    def get_all_img_list(self, max_imgs=None):
        """
        获取所有图片列表, 并从列表中随机挑选一定数量的图片作为数据集的图片列表
        """
        if not Path(self.label_path).exists():
            raise ValueError(f'label file {self.label_path} not exists')
        
        with open(self.label_path, 'r') as f:
            label_result = json.load(f)

        
        img_ids_all = [label['image'] for label in label_result if 'image' in label]

        if not img_ids_all:
            raise ValueError(f'label file {self.label_path} not contains any image')
        
        # concat root and ids
        img_ids_all = [str(self.image_path / img_id) for img_id in img_ids_all]

        # randomly select
        if max_imgs is not None:
            if max_imgs > len(img_ids_all):
                max_imgs = len(img_ids_all)
                print(f'Max images {max_imgs}')
            img_ids = random.sample(img_ids_all, max_imgs)
            print(f'Randomly select {len(img_ids)} images from {len(label_result)} images')
        else:
            img_ids = img_ids_all

        return img_ids
    
    def save(self, img_list):
        """
        将图片列表俄存到指定文件处
        """
        # save result
        with open(self.save_path, 'w') as f:
            json.dump(img_list, f, indent=2)
            print(f"Total samples: {len(img_list)}")
            print(f'Save img list to {self.save_path}')

    def __call__(self, max_imgs=None) -> Any:
        img_list = self.get_all_img_list(max_imgs)
        self.save(img_list)
        


if __name__ == "__main__":
    data_info_gen = DataInfoGen(label_path=label_path, save_path=save_path)
    data_info_gen(max_imgs=None)
