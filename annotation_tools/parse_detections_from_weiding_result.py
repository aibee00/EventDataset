""" 脚本说明

从weiding生成的detections结果中解析出bboxes，保存为json格式

weiding生成的detections结果text每一行的格式如下(包含多个bboxes):
image_name conf category rect_x rect_y rect_w rect_h child_category child_score conf category rect_x rect_y rect_w rect_h child_category child_score ...

json 格式：
[
    {
        "image": "volvo_jinan_xfh_20210617/20210617/ch01003_20210617133000/1500.jpg",
        "bbox": "person:[0.767, 0.223, 0.849, 0.59];person:[0.421, 0.062, 0.468, 0.282]",
        "dense_caption": "person:[0.767, 0.223, 0.849, 0.59];person:[0.421, 0.062, 0.468, 0.282]"
    },
    {
        "image": "volvo_jinan_xfh_20210617/20210617/ch01003_20210617133000/1512.jpg",
        "bbox": "person:[0.775, 0.232, 0.854, 0.592];person:[0.417, 0.068, 0.477, 0.289]",
        "dense_caption": "person:[0.775, 0.232, 0.854, 0.592];person:[0.417, 0.068, 0.477, 0.289]"
    },
    ...
]
"""

import argparse
import os
import json
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from annotation_tools.utils import H, W, normalize

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file", default="/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/result_new_4000.txt")
    parser.add_argument("--output_file", type=str, help="output file", default="/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/weiding_detections_4000.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(line.strip())
    
    output_path = os.path.dirname(args.output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    data_result = []

    for line in data:
        line = line.split(" ")
        img_name = os.path.basename(line[0])
        line = line[1:]
        assert len(line) % 8 == 0
        num = int(len(line) / 8)

        bboxes = []
        for i in range(num):
            cls = line[8 * i + 1]
            if cls != "1":
                continue
            xywh = line[8 * i + 2: 8 * i + 6]
            x1, y1, w, h = map(int, xywh)
            x2 = x1 + w
            y2 = y1 + h

            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)

        # Process img_name format
        site_id, sub_dir, name = img_name.split("__")
        if site_id == sub_dir:
            img_name = site_id + "/" + name
        else:
            date = site_id.split("_")[-1]
            img_name = site_id + "/" + date + "/" + sub_dir + "/" + name
        print(img_name)

        # 归一化
        bboxes = normalize(bboxes, H, W)

        # 把bboxes安装x1坐标排序一下
        bboxes = sorted(bboxes, key=lambda x: x[0])

        bboxes = ";".join(["person:" + str(bbox) for bbox in bboxes])
        data_dict = {"image": img_name, "bbox": bboxes}
        data_result.append(data_dict)

    # 打乱顺序
    # 设置随机数生成器的种子
    random.seed(42)
    random.shuffle(data_result)
        
    # save to json
    with open(args.output_file, encoding="utf-8", mode ="w") as f:
        json.dump(data_result, f, indent=4)
    
    print(f"Success! {args.output_file} saved")

