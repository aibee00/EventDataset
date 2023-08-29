""" Convert caption to vqa
"""
import os
import sys
import json
from tqdm import tqdm
import random
import cv2
from copy import deepcopy


input_file = sys.argv[1]  # label_train.json / label_test.json ...
output_file = sys.argv[2]  # vqa_train.json / vqa_test.json ...

IMG_H = 1440
IMG_W = 2560
IMG_SIZE = (IMG_H, IMG_W)

def convert_to_vqa(data):
    """ Convert caption to vqa
    """
    vqa = []
    for datum in tqdm(data):
        vqa.append({
            'question_id': datum['image_id'],
            'question': f"<context>{datum['dense_caption']}</context>{datum['prompt']}",
            'image_id': datum['image_id'],
            'answer': datum['caption'],
            'image': datum['image']
        })

    return vqa

def norm_coords(coords, img_size, round_bit=3):
    """ Normalize coordinates
    """
    h, w = img_size
    x1, y1, x2, y2 = coords
    x1 = round(x1 / w, round_bit)
    y1 = round(y1 / h, round_bit)
    x2 = round(x2 / w, round_bit)
    y2 = round(y2 / h, round_bit)
    return [x1, y1, x2, y2]


def dense_caption_to_dict(dense_caption):
    """ Convert dense caption string format to dict format
    e.g.: "man wearing a black shirt: [580, 351, 846, 988]; white suv parked on the street: [1429, 58, 1821, 342]; woman wearing black dress: [1080, 390, 1364, 980]; man wearing a white shirt and black pants: [870, 344, 1064, 901]; man wearing a black shirt and khaki shorts: [191, 307, 412, 854]; woman wearing pink shirt: [2229, 78, 2373, 412]; woman wearing a dress: [2058, 104, 2198, 468]; a woman wearing a black skirt and white shirt: [1943, 66, 2052, 395]; the floor is white: [0, 94, 2559, 1418]; the word urinoir on the bottom: [1970, 1345, 2331, 1434]; words on the building: [4, 0, 1172, 83]; black shoes on the womans feet: [1123, 892, 1362, 978]; the man is wearing a black and white shirt: [593, 445, 800, 678]; grey chair at table: [5, 1252, 360, 1433]; "
    """
    dense_caption = dense_caption.strip()
    captions = dense_caption.split(';')
    captions = [cap for cap in captions if cap != '']
    captions = [cap.split(':') for cap in captions]

    # Filtering out captions except for person or car
    keep_keywords = ["person", "car", "man", "woman"]
    captions = [cap for cap in captions if any(kw in cap[0].lower() for kw in keep_keywords)]

    dense_caption = {}
    for anno, coord in captions:
        anno = anno.strip()
        coord = eval(coord.strip())
        dense_caption[anno] = coord

    return dense_caption


def convert_to_vqa2(data, norm=True):
    """ Convert caption to vqa

    data: list(dict), dict k: 'image'/image_id/prompt/task/caption/dense_caption/box

    question: e.g. "What the coords of the 'man wearing black and white sneakers'. The format of output must like this: <box>[x1,y1,x2,y2]</box>"
    answer: "<box>[x1,y1,x2,y2]</box>"
    """
    for i, sample in enumerate(data):
        dense_caption = sample['dense_caption']
        dense_caption = dense_caption.strip()
        captions = dense_caption.split(';')
        captions = [cap for cap in captions if cap != '']
        captions = [cap.split(':') for cap in captions]
        
        # question answer
        anno, coords = random.choice(captions)

        # options
        options = [eval(coord.strip()) for anno, coord in captions]
        
        # normalize
        if norm:
            options = [norm_coords(coords, IMG_SIZE) for coords in options]

        # options = "["
        # for anno, coord in captions:
        #     options += str(coord.strip()) + ','
        # options = options[:-1]
        # options += ']'
        
        question = f"What are the bounding box coordinates of <{anno}>. Please select one of the following answers: "
        # question = f"Please caption the region<{coords.strip()}> in detail."
        
        # the answer is all bboxes for candidates
        norm_answer = list(norm_coords(eval(coords.strip()), IMG_SIZE))
        answers = [f"{norm_answer}"]
        # answers = f"{anno}"
    
        # add to orignal data
        sample['question'] = question  # Instruction
        sample['question_id'] = i
        sample['answer'] = answers
        sample['options'] = options

    return data


def convert_to_vqa3(data):
    """ Convert dense caption to dict format
    """
    vqa = []
    for datum in tqdm(data):
        dense_caption = datum['dense_caption']
        # to dict format
        dense_caption = dense_caption_to_dict(dense_caption)
        # update into data
        # datum['dense_caption'] = dense_caption

        for caption in dense_caption.items():
            vqa.append({
                'dense_caption': caption,
                'image': datum['image']
            })

    print(f"Total samples: {len(vqa)}")  # test: Total samples: 73373; train: Total samples: 693220
    return vqa


def convert_to_vqa4(data):
    """ Extract bbox from raw label json and convert to dict

    bbox format : "bbox": "person: [1104, 253, 1203, 441], potted plant: [1052, 50, 1207, 189], person: [783, 191, 873, 341], car: [1318, 194, 2119, 805], car: [0, 203, 146, 311], car: [1982, 150, 2396, 458], person: [130, 172, 202, 282], car: [0, 278, 667, 541], person: [892, 154, 1060, 434], potted plant: [1744, 651, 2483, 1387], car: [15, 434, 1497, 1418], person: [1624, 231, 1743, 337], potted plant: [2288, 400, 2558, 815], person: [760, 176, 881, 383], person: [725, 185, 835, 415], person: [1998, 274, 2248, 673], car: [900, 129, 1486, 425], person: [1028, 175, 1163, 435]",
    """
    vqa = []
    for datum in tqdm(data):
        bbox = datum['bbox']
        # if bbox is empty skip it
        if not bbox:
            continue

        # to dict format
        groups = bbox_filter(bbox, max_capacity=10000)
        if len(groups) > 1:
            print(f"Found more than 6 bboxes: {datum['image']}, {len(groups)} groups: {groups}")

        for group in groups:
            datum['bbox'] = group
            datum['image_size'] = IMG_SIZE
            vqa.append(deepcopy(datum))

    print(f"Total original samples: {len(data)}")
    print(f"Total samples after filter out sample with empty bbox: {len(vqa)}")
    
    return vqa


def convert_to_vqa5_llava(data):
    """ Extract bbox from raw label json and convert to dict

    bbox format : "bbox": "person: [1104, 253, 1203, 441], potted plant: [1052, 50, 1207, 189], person: [783, 191, 873, 341], car: [1318, 194, 2119, 805], car: [0, 203, 146, 311], car: [1982, 150, 2396, 458], person: [130, 172, 202, 282], car: [0, 278, 667, 541], person: [892, 154, 1060, 434], potted plant: [1744, 651, 2483, 1387], car: [15, 434, 1497, 1418], person: [1624, 231, 1743, 337], potted plant: [2288, 400, 2558, 815], person: [760, 176, 881, 383], person: [725, 185, 835, 415], person: [1998, 274, 2248, 673], car: [900, 129, 1486, 425], person: [1028, 175, 1163, 435]",

    目标label 3个keys: 
        1. id: image_id
        2. image: image_name
        3. conversations: List, [conversation1, conversation2, ...]

    # 设计几个问题L:
    1. How many people are there in the picture?
    2. Please specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
    """
    # conversation
    q1_human = {
        "from": "human",
        "value": "How many people are there in the picture?"
    }

    q2_human = {
        "from": "human",
        "value": "How many cars are there in the picture?"
    }

    q3_human = {
        "from": "human",
        "value": "Please specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y."
    }

    a_gpt = {
        "from": "gpt",
        "value": ""
    }

    vqa = []
    for datum in tqdm(data):
        bbox = datum['bbox']
        id = datum['image_id']
        image = datum['image']
        # if bbox is empty skip it
        if not bbox:
            continue

        # to dict format
        groups = bbox_filter(bbox, max_capacity=10000)
        if len(groups) > 1:
            print(f"Found more than 6 bboxes: {datum['image']}, {len(groups)} groups: {groups}")

        for group in groups:
            conversations = []

            # Conversation 1
            conversations.append(q1_human)
            num_persons = count_speify_objs(group, "person")
            a1 = a_gpt.copy()
            a1['value'] = str(num_persons)
            conversations.append(a1)

            # Conversation 2
            conversations.append(q2_human)
            num_cars = count_speify_objs(group, 'car')
            a2 = a_gpt.copy()
            a2['value'] = str(num_cars)
            conversations.append(a2)

            # Conversation 3
            conversations.append(q3_human)
            a3 = a_gpt.copy()
            a3['value'] = group
            conversations.append(a3)

            sample = {
                "id": id,
                "image": image,
                "conversations": conversations
            }

            vqa.append(sample)

    print(f"Total original samples: {len(data)}")
    print(f"Total samples after filter out sample with empty bbox: {len(vqa)}")
    
    return vqa


def count_speify_objs(bbox, obj_name="person"):
    """
    bbox: "person:[0.735, 0.179, 0.792, 0.505];car:[0.281, 0.542, 0.732, 0.988];person:[0.653, 0.14, 0.695, 0.417];person:[0.386, 0.172, 0.463, 0.526];car:[0.0, 0.256, 0.359, 0.708];person:[0.436, 0.113, 0.476, 0.306];person:[0.962, 0.231, 1.0, 0.358];person:[0.691, 0.166, 0.739, 0.439];person:[0.679, 0.158, 0.73, 0.451];person:[0.316, 0.116, 0.363, 0.351]"
    """
    # Count number of person in bbox
    objs = bbox.split(";")
    persons = [obj for obj in objs if obj_name in obj]
    return len(persons)


def bbox_filter(bbox, max_capacity = 10):
    """
    bbox: "person: [1104, 253, 1203, 441], potted plant: [1052, 50, 1207, 189], person: [783, 191, 873, 341], car: [1318, 194, 2119, 805], car: [0, 203, 146, 311], car: [1982, 150, 2396, 458], person: [130, 172, 202, 282], car: [0, 278, 667, 541], person: [892, 154, 1060, 434], potted plant: [1744, 651, 2483, 1387], car: [15, 434, 1497, 1418], person: [1624, 231, 1743, 337], potted plant: [2288, 400, 2558, 815], person: [760, 176, 881, 383], person: [725, 185, 835, 415], person: [1998, 274, 2248, 673], car: [900, 129, 1486, 425], person: [1028, 175, 1163, 435]"
    """
    key_types = set(['person', 'car'])  # to filter

    groups = []  # 如果一张图中有多于6个bbox，我们会分组，每隔6个分一组, 为了不超过max_txt_len

    bucket = []
    boxes = bbox.split('],')
    num = 0
    for item in boxes:
        name, coords = item.split(':')
        name = name.strip()
        coords = coords.strip()

        if not coords.endswith(']'):
            coords += ']'

        # Filter out key types
        if name not in key_types:
            # print(f"Skipping type : {name}")
            continue

        # 处理边界坐标值
        coords = eval(coords)
        coords = [0 if c < 0 else c for c in coords]
        
        # print(f"name: {name}, coords: {coords}")
        cur_box = f"{name}:{norm_coords(coords, IMG_SIZE)}"

        # 判断当前bucket是否已满
        if len(bucket) == max_capacity:
            # num置0, 当前bucket装入groups, 并清空bucket
            num = 0
            bucket_str = ";".join(bucket)  # 先把list转为str
            groups.append(bucket_str)
            bucket = []
            bucket.append(cur_box)  # 把当前box放入新桶
        else:
            bucket.append(cur_box)
            num += 1

    # 如果bucket还有剩余，则也装入groups
    if bucket:
        bucket_str = ";".join(bucket)
        groups.append(bucket_str)

    return groups


def convert_to_vqa6_onlycar(data):
    """ Filter caption about car only in the dense_captions of the data
    """
    vqa = []
    for item in tqdm(data):
        dense_caption = item['dense_caption']
        captions = dense_caption.split(';')
        captions = [caption for caption in captions \
                    if ' car ' in caption and 'tire' not in caption and 'people' not in caption and \
                        'woman' not in caption and 'man' not in caption and 'windshield' not in caption]
        
        if not captions:
            continue

        # dense_caption = ";".join(captions)
        # item['dense_caption'] = dense_caption
        # vqa.append(item)

        # 这里我们将多个描述split为多个样本用来增加训练样本数
        for caption in captions:
            new_item = item.copy()

            # 我们把caption中car的bbox坐标归一化一下
            desc, bbox = caption.split(':')
            bbox = eval(bbox)
            bbox = [0 if c < 0 else c for c in bbox]
            bbox = norm_coords(bbox, IMG_SIZE)
            caption = f"{desc}:{bbox}"

            new_item['dense_caption'] = caption.strip()
            vqa.append(new_item)

    print(f"Total original samples: {len(data)}")
    print(f"Total samples after filter out sample with car bbox: {len(vqa)}")
    return vqa


with open(input_file) as f:
    data = json.load(f)

with open(output_file, 'w') as f:
    # json.dump(convert_to_vqa2(data), f, indent=2)
    # json.dump(convert_to_vqa3(data), f, indent=2)
    # json.dump(convert_to_vqa4(data), f, indent=2)
    # json.dump(convert_to_vqa5_llava(data), f, indent=2)
    json.dump(convert_to_vqa6_onlycar(data), f, indent=2)
