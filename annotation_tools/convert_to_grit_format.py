""" 
GRiT format example:
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
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import os.path as osp

sys.path.append(osp.join(osp.dirname(__file__), '..'))

from tqdm import tqdm

from annotation_tools.convert_label_to_llava_format import Filter_NUM, merge_labels, parse_label_result_v1, plot_boxes_on_image_resave
from utils import H, W, denorm, xyxy2xywh


def denorm_boxes_in_caption(caption):
    """
    caption format example:
    [0.496, 0.472, 0.636, 0.974]这个人正从大门进入店内，他穿着深蓝色的带连体帽的长款羽绒服，黑色的裤子和黑色的耐克运动鞋，右手拿着一杯奶茶，背着黑色的双肩包，戴着白色的蓝牙耳机。他正沿着白色条纹的道路向店内走去，他的左前方圆形的汽车展台上摆放着一辆黑色的红旗轿车，轿车被金色的链子围档在中间。他的正前方站着男顾客[0.788, 0.356, 0.878, 0.798]，和男顾客[0.896, 0.54, 1.0, 0.991]，还有一个男店员。
    """
    # 获取所有'['字符在caption中的index
    cap = deepcopy(caption)
    while cap.find('[') != -1:
        start = cap.find('[')
        end = cap.find(']')
        if end == -1:
            break
        box_to_repl = cap[start:end+1]
        box_str = cap[start+1:end]
        box_str = box_str.replace(' ', '')
        box_list = box_str.split(',')
        try:
            box = list(map(float, box_list))
        except:
            print(f"box: {box_list}")
            import pdb; pdb.set_trace()
        box = denorm(box, H, W)
        box = xyxy2xywh(box)
        box = [int(i) for i in box]
        caption = caption.replace(box_to_repl, str(box))
        cap = cap[end+1:]
    return caption


def split_caption_to_box_caption(labels):
    """ 
    conversations : List(dict)
    bboxes : List(List(float))
    """
    captions = []
    bboxes = []
    captoins_ret = []
    for line in labels.split('\n'):
        if line == '':
            continue
        # 提取出每行开头的坐标，例如：[0.789, 0.474, 0.865, 0.976]
        line = line.strip()
        if line[0] != '[':
            continue

        captions.append(line)

        end = line.find(']')
        if end == -1:
            continue
        box_str = line[1:end]
        box_str = box_str.replace(' ', '')
        box_list = box_str.split(',')
        box = list(map(float, box_list))
        bboxes.append(box)

        captoins_ret.append(
            {
                "caption": line,
                "bbox": box
            }
        )
    
    assert len(captions) == len(bboxes), \
        f"Error! The length of captions({len(captions)}) is not match with bboxes({len(bboxes)}): captions: {captions}, bboxes: {bboxes}"

    return captoins_ret, captions, bboxes


def image_resave(img_path, save_path):
    """ Plot boxes on image and save it.
    Args:
        img_path: path of image
        boxes: boxes to be plotted
    Returns:
        None
    """
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)
    
    image_id = img_path.split('images/')[-1]
    img_path_des = Path(save_path) / image_id
    if not Path(img_path_des).parent.exists():
        Path(img_path_des).parent.mkdir(exist_ok=True, parents=True)
    
    # If this image has exist -> skip save it by overwrite
    if not Path(img_path_des).exists():
        print(f'saving image: {img_path_des}')
        # cv2.imwrite(img_path.as_posix(), img)
        shutil.copyfile(img_path, img_path_des)
        return
    
    print(f'image {img_path} already exist, skip saving')
    return


def convert_label_to_grit(label_path, new_label_path, label_path_v1=None, train_img_list_v1=None, images_save_path=None):
    with open(label_path, 'r') as f:
        labels = json.load(f)

    if label_path_v1 is not None:
        labels_v1 = parse_label_result_v1(label_path_v1, train_img_list_v1)
        # merge labels_v1 and labels
        labels = merge_labels(labels_v1, labels)

        with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
    
    # Parse original label and convert to llava: key: id,image,conversations
    labels_grit = {
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "object"
            }
        ],
        "images": []
    }

    num_boxes_all = 0
    for num, (idx, label) in enumerate(tqdm(labels.items())):
        if int(idx) > Filter_NUM:
            continue

        img = label['img']  # "img": "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01002_20210617125000/3204.jpg"
        image_id = img.split('images/')[-1]
        caption = label['caption'] if 'caption' in label else label['label']
        global_caption = label['global_caption']

        labels_grit['images'].append({
            "file_name": image_id,
            "height": H,
            "id": num + 1,
            "width": W
        })

        captions_ret, captions, boxes = split_caption_to_box_caption(caption)  # 每个box对应一个caption [{"caption": v, "bbox": v}]

        # plot_boxes_on_image_resave
        if Plot:=False:
            plot_boxes_on_image_resave(img, boxes, images_save_path)
        else:
            image_resave(img, images_save_path)

        num_boxes_all += len(boxes)

        # Gen annotations
        for item in captions_ret:
            _box = item['bbox']
            caption = item['caption']
            caption = denorm_boxes_in_caption(caption)
            box = xyxy2xywh(denorm(_box, H, W))
            box = [int(i) for i in box]

            i = len(labels_grit['annotations']) + 1
            
            labels_grit['annotations'].append({
                "area": box[2] * box[3],
                "bbox": box,
                "caption": caption,
                "category_id": 1,
                "id": i,
                "image_id": num + 1,
                "iscrowd": 0
            })

    with open(new_label_path, 'w', encoding='utf-8') as f:
        json.dump(labels_grit, f, ensure_ascii=False, indent=4)
    print(f'done! Saving into: {new_label_path}')
    print(f"Number of all of boxes: {num_boxes_all}")



if __name__ == "__main__":
    # convert_label_to_grit(
    #     label_path='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2.json',
    #     new_label_path='/training/wphu/Dataset/grit/label_result_all_grit_addimg.json',
    #     label_path_v1="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result.json",
    #     train_img_list_v1="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json",
    #     images_save_path='/training/wphu/Dataset/grit/images'
    # )

    # 英文版
    convert_label_to_grit(
        label_path='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json',
        new_label_path='/training/wphu/Dataset/grit/label_result_all_grit_en.json',
        label_path_v1="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_en.json",
        train_img_list_v1="/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list_v1.json",
        images_save_path='/training/wphu/Dataset/grit/images'
    )
    print('Done!')
