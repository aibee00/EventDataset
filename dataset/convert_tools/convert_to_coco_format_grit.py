""" 这个脚本说明:
基于convert_to_coco_format_grounding.py脚本
使用yolos为数据集中每个img检测出所有的物体的bbox作为背景信息添加到label中

注意！！！
这个脚本需要在docker内运行, 先启动docker:
`bash start_docker.sh`
registry.aibee.cn/aibee/eventgpt:grit.v1.0: 集成了grit的环境
"""

import os
import os.path as osp
import json
import sys
from tqdm import tqdm
import cv2
import numpy as np

from task_define import TASK

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

from grit_src.image_dense_captions import image_caption_api


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


def convert2(labels, detector=None, captionor=None, dense_caption_model=None):
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

        print(f"img: {abs_path}")

        if detector:
            # 使用检测器生成所有物体的bbox
            bboxes_context = detector.detect(abs_path)
            print(f"\t bboxes_context: {bboxes_context}")

        if captionor:
            # 使用图片描述器生成对图片的描述
            caption_blip2 = captionor.generate(abs_path)
            print(f"\t caption_blip2: {caption_blip2}")

        if dense_caption_model:
            dense_caption = dense_caption_model.generate(abs_path)
            print(f"\t dense_caption: {dense_caption}")

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
                        "task": task_token,
                        "bbox": bboxes_context,
                        "caption_blip2": caption_blip2,
                    })
                else:  # train
                    new_labels.append({
                        "image": relative_path,
                        "caption": "##".join(annotations), # str
                        "image_id": fname.split('.')[0],
                        "task": task_token,
                        "bbox": bboxes_context,
                        "caption_blip2": caption_blip2,
                        })
    
    return new_labels


class YolosObjectDetection():
    def __init__(self) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
        pass
        
    def get_results(self, img_path):
        """
        Args:
            img_path: path of img in local
        Returns:
            dict: dict(tuple): tuple(score, bbox)
        """
        detections = []

        image = Image.open(img_path)

        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label = self.model.config.id2label[label.item()]
            score = round(score.item(), 3)

            detections.append((label, score, box))

        # print detections
        # print(f"img: {img_path}, detections: {detections}")

        return detections
    
    @staticmethod
    def convert_context_description(detections):
        """ Convert detections into context description
        Args:
            detections: list(tuple): list(label, score, bbox)
        Returns:
            dict: context description
        """
        if not detections:
            return ""
        
        # convert to string format: "person: box, car: box, ..."
        context_str = ""
        for label, score, box in detections:
            context_str += f"{label}: {box}, "

        # 去掉最后的两个字符: ", "
        context_str = context_str[:-2]

        return context_str
    
    def detect(self, img_path):
        """
        Args:
            img_path: path of img in local
        Returns:
            dict: dict(tuple): tuple(score, bbox)
        """
        results = self.get_results(img_path)
        context = self.convert_context_description(results)

        return context
    

class Blip2Captionor():
    def __init__(self) -> None:
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.model.to(self.device)

    def generate(self, img_path):
        """
        Args:
            img_path: str
        Returns:
            str: caption
        """
        device = self.device

        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DenseCaptioning():
    def __init__(self, device):
        self.device = device


    def initialize_model(self):
        pass

    def image_dense_caption_debug(self, image_src):
        dense_caption = """
        1. the broccoli is green, [0, 0, 333, 325]; 
        2. a piece of broccoli, [0, 147, 143, 324]; 
        3. silver fork on plate, [4, 547, 252, 612];
        """
        return dense_caption
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_api(image_src, self.device)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("Step2, Dense Caption:\n")
        print(dense_caption)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return dense_caption


if __name__ == '__main__':
    # save
    # new_labels = convert(labels)

    # detector = YolosObjectDetection()
    # captionor = Blip2Captionor()
    detector = None
    captionor = None

    if torch.cuda.is_available():
        dense_caption_device = "cuda:2"
    else:
        dense_caption_device = "cpu"
    dense_caption_model = DenseCaptioning(device=dense_caption_device)

    new_labels = convert2(labels, dense_caption_model=dense_caption_model)
    print(f"Total valid samples: {len(new_labels)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)


