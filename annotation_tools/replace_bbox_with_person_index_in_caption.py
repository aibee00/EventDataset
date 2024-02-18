""" 说明
该脚本目的是将标注中的`bbox`转为`人物代号`的方式来描述。
"""
import json
import argparse
from pathlib import Path

from tqdm import tqdm

from utils import denorm, get_label_info, plot_bboxes_on_image, H, W
import cv2
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_result_v1", type=str, default="/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_train.json")
    parser.add_argument("--detection_result_v2", type=str, default="/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/weiding_detections_4000.json")
    parser.add_argument("--label_result_v1", type=str, default="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_merged.json")
    parser.add_argument("--label_result_v2", type=str, default="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json")
    parser.add_argument("--save_path", type=str, default="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/person_index/label_result_v1v2_person_index.json")
    parser.add_argument("--original_image_path", type=str, default="/training/wphu/Dataset/lavis/eventgpt/images/")
    parser.add_argument("--image_save_path", type=str, default="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/person_index/images/")

    return parser.parse_args()


class Bbox2PersonIndex:
    def __init__(self, detection_result, label_result, save_path=None):
        self.detection_result = detection_result
        self.label_result = label_result
        self.save_path = save_path

        self.detection_map = self._get_detection_map()  # image_id to detections

    @staticmethod
    def merge(labels_v1, labels_v2):
        label_result = {}
        for i, (idx, data) in enumerate(tqdm(labels_v1.items())):
            label_result[str(-(i + 1))] = data
        label_result.update(labels_v2)
        return label_result

    def _get_detection_map(self, ):
        labels = json.loads(open(self.detection_result, 'r').read())
        labels.sort(key=lambda x: x['image'])
        detection_map = {label['image']: label for label in labels}
        return detection_map

    def _get_bbox_to_index_map(self, image_path):
        use_relative_path = False if next(iter(self.detection_map)).startswith('/') else True
        bboxes_norm = get_label_info(image_path, self.detection_map, "bbox", use_relative_path=use_relative_path)
        bboxes_norm.sort(key=lambda x: x[0])
        bbox_to_index_map = {}
        for i, bbox in enumerate(bboxes_norm):
            bbox_to_index_map[tuple(bbox)] = i
        return bbox_to_index_map
    
    def _replace_bbox_with_index(self, sample_label, bbox_to_index_map, english=True):
        """ Replace bbox with index
        Args:
            label (str): label string
            bbox_to_index_map (dict): bbox to index map
        Returns:
            str: label string with bbox replaced with index
        """
        sample_label = sample_label.strip().replace('\n\n', '\n')

        labels_new = []
        for cap_per_box in sample_label.split('\n'):
            if cap_per_box.strip() == "":
                continue
            if cap_per_box.find(']') == -1:
                labels_new.append(cap_per_box)
                continue

            label = cap_per_box.strip()
            cap_begin = label.find(']')
            label_prefix = label[:cap_begin + 1]
            label_suffix = label[cap_begin + 1:]
            for bbox, index in bbox_to_index_map.items():
                person_num = f"(Person_{index})" if english else f"({index}号人物)"
                label_suffix = label_suffix.replace(str(list(bbox)), person_num)
            label_new = label_prefix + label_suffix
            labels_new.append(label_new)
        return '\n\n'.join(labels_new)

    def convert(self):
        with open(self.label_result, "r") as f:
            label_result = json.load(f)

        for idx, data in tqdm(label_result.items()):
            caption_ch = data['label']

            if 'caption' not in data:  # 没标注英文的caption
                continue

            caption_en = data['caption']  # caption is en caption, label is ch caption
            image_path = data['img']  # e.g. /training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500/1140.jpg

            # get bbox to index map
            bbox_to_index_map = self._get_bbox_to_index_map(image_path)
            # replace bbox with index
            caption_ch = self._replace_bbox_with_index(caption_ch, bbox_to_index_map, english=False)
            caption_en = self._replace_bbox_with_index(caption_en, bbox_to_index_map, english=True)

            data['label'] = caption_ch
            data['caption'] = caption_en

        return label_result
    



class BBoxPlotter():
    def __init__(self) -> None:
        pass

    @staticmethod
    def parse_bboxes_from_detection(detection):
        """
        "bbox": "person:[0.0, 0.628, 0.091, 0.996];person:[0.0, 0.485, 0.048, 0.855];person:[0.057, 0.491, 0.184, 0.992];person:[0.129, 0.353, 0.239, 0.874];person:[0.504, 0.12, 0.561, 0.328];person:[0.576, 0.13, 0.649, 0.33]"
        """
        bboxes_str = detection['bbox']
        bboxes_str = bboxes_str.replace('person:', '')
        bboxes = [box for box in bboxes_str.split(';') if box]
        bboxes_norm = []
        for bbox in bboxes:
            bbox = bbox.replace('[', '')
            bbox = bbox.replace(']', '')
            bbox = bbox.split(',')
            bbox = list(map(float, bbox))
            bboxes_norm.append(bbox)
        return bboxes_norm

    @staticmethod
    def _get_new_img_path(image_path, new_image_dir):
        """ 从原始label中的img_path转换为新的img_name形式
        Input:
            image_path: 原文件路径 Path Like: "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500/1140.jpg"
            new_image_dir: 新文件目录，默认为None，则为保存到self.save_dir/images下。Path Like: '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images'

        Return:
            img_path: Path Like: '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images/volvo_jinan_xfh_20210617_ch01019_20210617172500_1140.jpg'
        
        Note:
            old img_path: 指 label 中的image_path形式
        """
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        if not isinstance(new_image_dir, Path):
            new_image_dir = Path(new_image_dir) if new_image_dir is not None else None
        image_dir = image_path.parent
        img_name = image_path.name
        sub_paths = image_dir.as_posix().split('/')
        new_img_path = new_image_dir / f'{sub_paths[-3]}__{sub_paths[-1]}__{img_name}'
        return new_img_path


    def plot_detections_on_image(self, detections_path_v1, detections_path_v2, label_result_path, image_dir, save_dir):
        detections = json.loads(open(detections_path_v1, 'r').read())
        detections.extend(json.loads(open(detections_path_v2, 'r').read()))

        det_map = {}
        for detection in detections:
            is_relative_path = False if detection['image'].startswith('/') else True
            image_path = detection['image'] if not is_relative_path else os.path.join(image_dir, detection['image'])
            detection['image'] = image_path
            det_map[image_path] = detection

        labels = json.loads(open(label_result_path, 'r').read())
        for idx, label in tqdm(labels.items(), desc='[detections]'):
            detection = det_map.get(label['img'])
            if detection is None:
                continue
            
            # Get full image path
            image_abs_path = detection['image']
            new_img_path = self._get_new_img_path(image_abs_path, save_dir)
            print(f"new_img_path: {new_img_path}")

            # Get bounding box info
            bboxes_norm = self.parse_bboxes_from_detection(detection)
            bboxes_norm.sort(key=lambda x: x[0])  # Sort bboxes by x coordinate
            bboxes = denorm(bboxes_norm, H, W)

            if True or not new_img_path.exists():
                img = plot_bboxes_on_image(image_abs_path, bboxes)
                cv2.imwrite(new_img_path.as_posix(), img)
        print(f"Detections saved to {save_dir}")
    
            
if __name__ == "__main__":
    args = parse_args()
    
    bbox2index = Bbox2PersonIndex(args.detection_result_v1, args.label_result_v1)
    label_result_v1 = bbox2index.convert()

    bbox2index = Bbox2PersonIndex(args.detection_result_v2, args.label_result_v2)
    label_result_v2 = bbox2index.convert()

    # Merge
    label_result_merged = bbox2index.merge(label_result_v1, label_result_v2)

    # Save result
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(label_result_merged, f, indent=4, ensure_ascii=False)
    print(f"Result saved: {args.save_path}")

    # Plot detections on image
    if Plot:= False:
        os.makedirs(os.path.dirname(args.image_save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt = BBoxPlotter()
        plt.plot_detections_on_image(
            args.detection_result_v1, 
            args.detection_result_v2,
            args.save_path, 
            args.original_image_path, 
            args.image_save_path
        )  # detection_result_v1: 1138714 张图, v2是从v1中采样的
