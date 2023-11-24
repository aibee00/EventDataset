""" 
vg dataset format example:
{
    'annotations': [
        {
            "area": 239475,  # area = w * h
            "bbox": [
                142,  # x
                665,  # y
                309,  # w
                775   # h
            ],
            "caption": "[0.506, 0.187, 0.571, 0.531] 这是一个穿着黑色长袖外套、黑色长裤和黑色休闲鞋的男士。他戴着白色口罩，左边紧靠着反光墙，右边是一位女士[0.542, 0.209, 0.625, 0.548]，他们站在展台前，正看着前方的黑色展车。可推断他和女士[0.542, 0.209, 0.625, 0.548]是一个组。根据穿着和行为可推断他可能是一名顾客。",
            "category_id": 1,
            "id": 1,
            "image_id": 1,
            "iscrowd": 0
        }, ....
    ], 
    'categories': [
        {
        "id": 1,
        "name": "object"
        }
    ],
    'images': [
        {
            "file_name": "GACNE_guangzhou_xhthwk_ch01001_20210717100142.jpg",
            "height": 1440,
            "id": 1,
            "width": 2560
        },...
    ]
}
"""

import json

from tqdm import tqdm



def xywh2xyxy(bbox):
    """
    Convert bbox from xywh to xyxy
    """
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def normlize_bbox(bbox, H, W, decimal=3):
    bbox[0] = round(bbox[0] / W, decimal)
    bbox[1] = round(bbox[1] / H, decimal)
    bbox[2] = round(bbox[2] / W, decimal)
    bbox[3] = round(bbox[3] / H, decimal)
    return bbox


def parse_vg_dataset_bbox_as_unit(label_path):
    """ Parse labels from vg dataset
    Retrun:
    {
        "image_id": index,
        "image": image relative path,
        "height": image_H,
        "width": image_W,
        "bbox_id": box_id,
        "bbox": bbox,  # 格式: normed [x1,y1,x2,y2]
        "caption": caption # 格式: normed [x1,y1,x2,y2] description
    }
    """
    dataset = json.loads(open(label_path, 'r').read())
    labels = dataset['annotations']

    # get image_name based on image_id
    images_map = {image['id']: image for image in dataset['images']}
    for item in labels:
        item['image'] = images_map[item['image_id']]['file_name']
        item['height'] = images_map[item['image_id']]['height']
        item['width'] = images_map[item['image_id']]['width']

        # convert bbox to xyxy
        H = images_map[item['image_id']]['height']
        W = images_map[item['image_id']]['width']
        bbox = normlize_bbox(xywh2xyxy(item['bbox']), H, W)
        item['bbox'] = bbox
        item['bbox_id'] = item['id']
        item['caption'] = f"{bbox} {item['caption']}"
    return labels


def parse_vg_dataset_image_as_unit(label_path):
    """ Parse labels from vg dataset, 基于图片的，caption是图片中所有bboxes的分别描述
    Retrun:
    {
        "image_id": index,
        "image": image relative path,
        "height": image_H,
        "width": image_W,
        "bbox_id": [box_id],
        "bbox": bboxes,  # 格式: normed [[x1,y1,x2,y2],...]
        "caption": caption # 格式: “[x1,y1,x2,y2] description \n [x1,y1,x2,y2] description ...”
    }
    """
    dataset = json.loads(open(label_path, 'r').read())
    annotations = dataset['annotations']
    images = dataset['images']
    labels = []

    img_anno_map = {}
    for anno in annotations:
        img_anno_map.setdefault(anno['image_id'], []).append(anno)

    for item in images:
        label = {}

        # Image part
        label['image_id'] = item['id']
        label['image'] = item['file_name']
        label['height'] = item['height']
        label['width'] = item['width']

        # Bboxe part
        all_bbox_annotations = img_anno_map[item['id']]
        bboxes = []
        bbox_ids = []
        captions = []
        for anno in all_bbox_annotations:
            bbox = normlize_bbox(xywh2xyxy(anno['bbox']), item['height'], item['width'])
            bboxes.append(bbox)
            bbox_ids.append(anno['id'])
            captions.append(f"{bbox} {anno['caption']} ")
        caption = "\n".join(captions)

        label['bbox_id'] = bbox_ids
        label['bbox'] = bboxes
        label['caption'] = caption

        labels.append(label)

    return labels


def convert_to_llava_format(label_path, new_label_path, gen_caption_based_bbox=False):
    """ Convert vg dataset to llava format
    Args:
        label_path: vg dataset path
        new_label_path: new llava dataset path
    Return:
        None
    """
    if gen_caption_based_bbox:
        labels = parse_vg_dataset_bbox_as_unit(label_path)
    else:
        labels = parse_vg_dataset_image_as_unit(label_path)

    # Parse original label and convert to llava: key: id,image,conversations

    # Prompt
    Instructions = f"Based on the provided image and corresponding bounding box coordinates, please offer a detailed description of the individual within the area. " + \
                f"These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y."

    labels_llava = []
    for idx, label in enumerate(tqdm(labels)):
        image_id = label['image']
        caption = label['caption']
        bbox = label['bbox']

        # 创建多轮对话过程
        conversations = [
            {
                "from": "human",
                "value": '<image>\n' + Instructions + '\n' + f'<bbox>: {bbox}\n'
            },
            {
                "from": "gpt",
                "value": caption
            }
        ]  # 初始化第一轮对话，加入context

        labels_llava.append({
            "id": idx,
            "image": image_id,
            "conversations": conversations
        })
    
    with open(new_label_path, 'w', encoding='utf-8') as f:
        json.dump(labels_llava, f, ensure_ascii=False, indent=4)
    print(f'done! Saving into: {new_label_path}')
    print(f"Number of all of boxes: {len(labels)}")



if __name__ == "__main__":
    use_based_on_each_bbox_description = False

    convert_to_llava_format(
        label_path="/training/wphu/Dataset/vg/annotations/train.json",
        new_label_path="/training/wphu/Dataset/vg/annotations/train_llava_based_on_image.json",
        gen_caption_based_bbox=use_based_on_each_bbox_description
    )

