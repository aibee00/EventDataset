import json
import os
import shutil
from tqdm import tqdm
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotation_tools.utils import H, W, denorm, get_label_info, plot_bboxes_on_image

WITH_BBOX = True
VERSION = "v2"

# 指定json文件和保存路径
# img_list_path = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json"
img_list_path = f"/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list_{VERSION}.json"

if WITH_BBOX:
    # save_path = "/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/images"
    save_path = f"/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/images_{VERSION}"
else:    
    save_path = "/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/images_raw"

# label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_onlyperson_train.json"
# label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_train.json"
label_path = "/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/weiding_detections_4000.json"

labels = json.loads(open(label_path, 'r').read())
labels.sort(key=lambda x: x['image'])

# convert to dict, key is image_id, value is label_infos
label_map = {label['image']: label for label in labels}

# 读取json文件以获取图片路径列表
with open(img_list_path, 'r') as f:
    img_list = json.load(f)

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"img_list: {len(img_list)}")
print(f"img_list set: {len(set(img_list))}")


def add_bboxes_to_img(img_path, label_map):
    # Get bounding box info
    try:
        bboxes_norm = get_label_info(img_path, label_map, "bbox")
    except:
        print(f"Error: {img_path} does not have bbox info, we will ignore it!!!")
        return cv2.imread(img_path)
    
    bboxes_norm.sort(key=lambda x: x[0])  # Sort bboxes by x coordinate
    bboxes = denorm(bboxes_norm, H, W)
    img = plot_bboxes_on_image(img_path, bboxes)
    return img


# 遍历图片路径列表并复制每个图片
for img_path in tqdm(img_list):
    # 获取子文件夹名称
    subfolder_name = img_path.split("/")[7]

    ch_name = img_path.split("/")[-2]
    
    # 从路径中获取图片文件名
    img_file_name = os.path.basename(img_path)
    
    # 在文件名前添加子文件夹名称作为前缀
    new_img_file_name = f"{subfolder_name}__{ch_name}__{img_file_name}"
    
    # 指定新的保存路径
    new_img_path = os.path.join(save_path, new_img_file_name)

    if WITH_BBOX:
        img = add_bboxes_to_img(img_path, label_map)
    
    # 执行复制操作
    if os.path.exists(new_img_path):
        print(f"Image {new_img_path} already exists, skipping...")
        continue

    if WITH_BBOX:
        cv2.imwrite(new_img_path, img)
    else:
        shutil.copy(img_path, new_img_path)

print(f"All images have been copied to {save_path}")
