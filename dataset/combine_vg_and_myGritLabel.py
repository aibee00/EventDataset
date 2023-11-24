""" 说明
vg: Visual Genome Dataset https://homes.cs.washington.edu/~ranjay/visualgenome/index.html
myGritLabel: 我收集的业务数据，用GRiT推断得到了bbox和caption.

Combine 的目的是将两个数据集合并，并且生成一个新的总数据集，用于预训练。
vg 数据量只用 trainset 部分，大概 7.7万 张图。
myGritLabel 数据量大概是 2.3万 张有效图，原集合有8万张，过滤掉 area < 0.01 的bbox后剩下2.3万。
"""

import json
import os.path as osp
import argparse

from pathlib import Path
from tqdm import tqdm


def args_parser():
    parser = argparse.ArgumentParser(description='Combine vg and myGritLabel')
    parser.add_argument('--vg_data_path', type=str, default='/training/wphu/Dataset/vg/annotations/vg_for_lavis_train.json')
    parser.add_argument('--myGritLabel_data_path', type=str, \
                        default='/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_llava_style_box_caption_train.json')
    parser.add_argument('--output_data_path', type=str, default='/training/wphu/Dataset/lavis/eventgpt/annotations/')
    args = parser.parse_args()
    return args


def preprocess_of_vg(vg_data):
    """ 
    vg 数据集处理，将图片路径替换为新的相对路径. 例如 51.jpg -> vg/images/51.jpg
    """
    for item in tqdm(vg_data):
        item['image'] = f"vg/images/{item['image']}"
    return vg_data


def count_num_of_image(data):
    """ 统计 vg 数据集中图片的数量 """
    image_set = set()
    for item in data:
        image_set.add(item['image'])
    num = len(image_set)
    print(f"Total image: {num}")
    return num


if __name__ == '__main__':
    args = args_parser()
    vg_data_path = args.vg_data_path
    myGritLabel_data_path = args.myGritLabel_data_path
    output_data_path = args.output_data_path

    vg_data_path = Path(vg_data_path)
    myGritLabel_data_path = Path(myGritLabel_data_path)
    output_data_path = Path(output_data_path)

    print(f"Loading vg data ...")
    vg_data = json.loads(open(vg_data_path, 'r').read())
    print(f"Loading myGritLabel data ...")
    myGritLabel_data = json.loads(open(myGritLabel_data_path, 'r').read())

    # 统计图片数量
    num_img_vg = count_num_of_image(vg_data)
    num_img_myGritLabel = count_num_of_image(myGritLabel_data)
    num_img_total = num_img_vg + num_img_myGritLabel

    print(f"Total vg data: {len(vg_data)} boxes, {num_img_vg} images")
    print(f"Total myGritLabel data: {len(myGritLabel_data)} boxes, {num_img_myGritLabel} images")

    vg_data = preprocess_of_vg(vg_data)

    # 将 myGritLabel 数据集合并到 vg 数据集中, 并保存为新的数据集.
    vg_data.extend(myGritLabel_data)

    save_path = output_data_path / 'vg_and_myGritLabel_train.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vg_data, f, indent=2, ensure_ascii=False)

    print(f"Result saved into: {save_path}")
    print(f"Total data: {len(vg_data)} boxes, {num_img_total} images.")



