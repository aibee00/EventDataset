import os
import os.path as osp
import json
import sys
from tqdm import tqdm
import cv2
import numpy as np

from task_define import TASK

input_file = sys.argv[1]  # e.g. label_vw.json or multi-json sep with ',', like label_vw.json,label_hongqi.json
output_file = sys.argv[2]  # e.g. label_train.json
is_val=sys.argv[3]  # is_val=1 or is_val=0

if is_val == 1:
    is_val = True
else:
    is_val = False

if ',' not in input_file:
    with open(input_file, 'r') as f:
        labels = json.load(f)
else:
    input_files = input_file.split(',')
    labels = {}
    for input_file in input_files:
        with open(input_file, 'r') as f:
            labels.update(json.load(f))

def convert(labels):
    new_labels = []
    for i, fname in enumerate(tqdm(labels)):
        label = labels[fname]

        if i == 0:
            print(label.keys())

        abs_path = label["img"]

        site_id = abs_path.split('/')[-3]
        relative_path = osp.join(site_id, fname)

        annotations = []
        annotations.append("描述:" + label["annotation"])
        annotations.append("背景信息:" + label["context"])
        if label.get("Car_context"):
            annotations[-1] += ";" + label["Car_context"]
        if label.get("Door_context"):
            annotations[-1] += ";" + label["Door_context"]

        if is_val:
            new_labels.append({
                "image": relative_path,
                "caption": annotations,  # list
                "image_id": fname.split('.')[0] 
            })
        else:
            new_labels.append({
                "image": relative_path,
                "caption": "##".join(annotations), # str
                "image_id": fname.split('.')[0]
            })
    return new_labels



def process_context(context, H, W):
    """ For grounding task
        [1] Split context infos and save into dict;
        [2] Convert string into list for every coords;
        [3] Convert coords ratio into real coords pixel by H,W
    Return:
        Dict: key: person_id, value: coords
    """
    if not context:
        return {}
    
    context_dict = {}
    for anno in context.split(';'):
        person_id = anno.split(':')[0]
        try:
            coords = anno.split(':')[1]
        except:
            import pdb; pdb.set_trace()

        # convert coords ratio into real coords pixel by H,W
        coords = eval(coords)

        if coords:
            coords = [int(coords[0] * W), int(coords[1] * H), int(coords[2] * W), int(coords[3] * H)]

        context_dict[person_id] = coords

    return context_dict


def process_door_context(door_context, H, W):
    """ Extract infos from str caption of door

    Input: 
        door_context: str: door catpion
        H: img height
        W: img width

    Return: Dict: key: door_id, value: door_info
    """
    if not door_context:
        return {}
    
    door_dict = {}
    for anno in door_context.split(';'):
        door_id = anno.split(':')[0]
        try:
            door_info = anno.split(':')[1]
        except:
            import pdb; pdb.set_trace()

        door_info = eval(door_info)

        if door_info:
            # convert coords ratio into real coords pixel by H,
            for point in door_info:
                point[0] = int(point[0] * W)
                point[1] = int(point[1] * H)

                # 截断point超出image的部分
                point[0] = max(min(point[0], W), 0)
                point[1] = max(min(point[1], H), 0)
                point[0] = min(max(point[0], 0), W)
                point[1] = min(max(point[1], 0), H)

            # 如果door两端点距离小于3则舍弃
            start = door_info[0]
            end = door_info[-1]
            if np.linalg.norm(np.array(start) - np.array(end)) < 3:
                door_info = []

        door_dict[door_id] = door_info

    return door_dict


def process_car_context(car_context, H, W):
    """ Extract infos from str caption of car

    Input: 
        car_context: str: car catpion
        H: img height
        W: img width

    Return: Dict: key: car_id, value: car_info
    """
    if not car_context:
        return {}
    
    car_dict = {}
    for anno in car_context.split(';'):
        car_id = anno.split(':')[0]
        try:
            car_info = anno.split(':')[1]
        except:
            import pdb; pdb.set_trace()

        car_info = eval(car_info)

        if car_info:
            # convert coords ratio into real coords pixel by H,
            for point in car_info:
                point[0] = int(point[0] * W)
                point[1] = int(point[1] * H)

                # 截断point超凿image的部分
                point[0] = max(min(point[0], W), 0)
                point[1] = max(min(point[1], H), 0)
                point[0] = min(max(point[0], 0), W)
                point[1] = min(max(point[1], 0), H)

        car_dict[car_id] = car_info

    return car_dict


def get_task(caption, use_index=False):
    """ Parse task from caption
    """
    task = ""
    if "接待" in caption:
        task = "INDIVIDUAL_RECEPTION"
    elif "访问Car" in caption:
        task = "CAR_VISIT"
    elif "该店大门" in caption:
        task = "STORE_INOUT"
    elif "批次" in caption:
        task = "COMPANION"
    elif "车Car" in caption:
        task = "CAR_INOUT"
    else:
        task = "UNKNOWN"

    if use_index:
        task = TASK[task]
    
    return task


def convert2(labels):
    """ For grounding task
        [1] 加入边界符
        [2] 直接把Bbox坐标嵌入到描述的<box> </box>中
        [3] 加入[TASK]特殊字符表示任务(表示不同的事件)
        [4] 将各个事件的描述分开为单独的pair训练。 
    """
    new_labels = []
    H, W = 0, 0
    for i, fname in enumerate(tqdm(labels)):
        label = labels[fname]

        if i == 0:
            print(label.keys())

        abs_path = label["img"]

        site_id = abs_path.split('/')[-3]
        relative_path = osp.join(site_id, fname)

        # 读取第一个image,获取image的H,W
        if i == 0:
            img = cv2.imread(abs_path)
            H, W = img.shape[:2]

        # Split each event annotation as a single caption
        captions = []
        for anno in label["annotation"].split(';'):
            captions.append(anno)

        # process context
        context = process_context(label["context"], H, W)

        # process door context
        if "Door_context" in label:
            door_context = process_door_context(label["Door_context"], H, W)
        else:
            door_context = {}

        # process car context
        if "Car_context" in label:
            car_context = process_car_context(label["Car_context"], H, W)
        else:
            car_context = {}

        # Generate one-pair for each caption
        for caption in captions:
            abandon = False

            # 将caption的person替换为坐标
            for person_id in context:
                # 如果person_box为空，则舍弃该样本
                person_box = context[person_id]
                if not person_box:
                    abandon = True
                    break

                # 将person_id替换为它的坐标信息
                person_info = f"{person_id}<box>{str(context[person_id])}</box>"
                # 去掉person_info中的空格
                person_info = person_info.replace(" ", "")
                caption = caption.replace(person_id, person_info)

            # 将Door_id替换为它的坐标信息
            for door_id in door_context:
                door_box = door_context[door_id]
                if not door_box:
                    abandon = True
                    break

                # 将person_id替换为它的坐标信息
                door_info = f"{door_id}<box>{str(door_context[door_id])}</box>"
                # 去掉door_info中的空格
                door_info = door_info.replace(" ", "")
                caption = caption.replace(door_id, door_info)

            # 将Car_id替换为它的坐标信息
            for car_id in car_context:
                car_box = car_context[car_id]
                if not car_box:
                    abandon = True
                    break

                # 将car_id替换为它的定位信息
                car_info = f"{car_id}<box>{str(car_context[car_id])}</box>"
                # 去掉car_info中的空栍
                car_info = car_info.replace(" ", "")
                caption = caption.replace(car_id, car_info)


            # 如果caption中有原始pid则舍弃
            if "<p-" in caption:
                abandon = True
            elif "<s-" in caption:
                abandon = True

            # Get task of different event
            use_index = True
            task = get_task(caption, use_index)
            task_token = "[TASK:{:2d}]".format(task) if use_index else f"[TASK:{task}]"
            # caption = task_token + "###" + caption

            annotations = []
            annotations.append(caption)

            if not abandon:
                if is_val:  # val
                    new_labels.append({
                        "image": relative_path,
                        "caption": annotations,  # list
                        "image_id": fname.split('.')[0],
                        "task": task_token
                    })
                else:  # train
                    new_labels.append({
                        "image": relative_path,
                        "caption": "##".join(annotations), # str
                        "image_id": fname.split('.')[0],
                        "task": task_token
                        })
    
    return new_labels


if __name__ == '__main__':
    # save
    # new_labels = convert(labels)
    new_labels = convert2(labels)
    print(f"Total valid samples: {len(new_labels)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)


