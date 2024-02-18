import json
from multiprocessing import Pool
import cv2
from pathlib import Path
import argparse
import os

from tqdm import tqdm


from utils import H, W, denorm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images/')
    parser.add_argument('--detections_json_path', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json')
    parser.add_argument('--image_save_path', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images_with_bboxes/')
    parser.add_argument('--num_proc', type=int, default=8)
    args = parser.parse_args()
    return args


def plot_bbox(params):
    image_path, bboxes, save_path = params
    if Path(save_path).exists():
        print(f"Skip: {save_path}")
        return
    img = cv2.imread(image_path)
    for i, bbox in enumerate(bboxes):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    print(f"Plotting: {save_path}")
    if not Path(save_path).exists():
        Path(save_path).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_path, img)
    return


class Plotter():
    def __init__(self, args) -> None:
        self.args = args

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
        if '__' in img_name:
            return new_image_dir / img_name
        new_img_path = new_image_dir / f'{sub_paths[-3]}__{sub_paths[-1]}__{img_name}'
        return new_img_path

    def plot_and_save(self, ):
        with open(self.args.detections_json_path, 'r') as f:
            detections = json.load(f)
        
        params = []
        for detection in detections:
            image_path = detection['image']
            # Get new image path
            new_img_path = self._get_new_img_path(image_path, self.args.image_save_path)

            if not detection['bbox']:
                # Copy raw image to the new image path
                print(f"Copying raw image {image_path} to {new_img_path.as_posix()}")
                os.system(f"cp {image_path} {new_img_path.as_posix()}")
                continue

            bboxes_norm = self.parse_bboxes_from_detection(detection)
            bboxes_norm.sort(key=lambda x: x[0])  # Sort bboxes by x coordinate
            bboxes = denorm(bboxes_norm, H, W)

            params.append((image_path, bboxes, new_img_path.as_posix()))

        if self.args.num_proc <= 1:
            for param in tqdm(params, desc='[plot_and_save]'):
                plot_bbox(param)  # Plot and save bboxes on image
        else:
            with Pool(self.args.num_proc) as pool:
                pool.map(plot_bbox, params)


if __name__ == '__main__':
    args = parse_args()
    plotter = Plotter(args)
    plotter.plot_and_save()
