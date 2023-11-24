import json
from pathlib import Path
import cv2
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from annotation_tools.utils import denorm


Filter_NUM = 413
USE_EN = True
MaxNum_forEval = 425


def parse_bboxes_from_caption(caption):
    """
    caption: str
    return: list of bboxes
    "[0.506, 0.187, 0.571, 0.531] 这是一个穿着黑色长袖外套、黑色长裤和黑色休闲鞋的男士。他戴着白色口罩，左边紧靠着反光墙，右边是一位女士[0.542, 0.209, 0.625, 0.548]，他们站在展台前，正看着前方的黑色展车。可推断他和女士[0.542, 0.209, 0.625, 0.548]是一个组。根据穿着和行为可推断他可能是一名顾客。\n\n[0.542, 0.209, 0.625, 0.548] 这是一个穿着黄绿相间并带有卡通图案的长袖上衣、白色短裙和白色休闲鞋的女士。她左手挽着一位男士[0.542, 0.209, 0.625, 0.548]，他们站在展台前，正看着前方的黑色展车。可推断她和男士[0.506, 0.187, 0.571, 0.531]是一个组。根据穿着和行为可推断她可能是一名顾客。\n\n[0.621, 0.377, 0.71, 0.874] 这是一个穿着黑白色的长袖羽绒服、黑色长裤和深色休闲鞋的男士。他手里拿着一个手机，正站在画面中央看手机。根据穿着和行为可推断他可能是一名顾客。\n\n[0.657, 0.766, 0.775, 0.997] 这是一个穿着黑色上衣的女士，她留着长发，在画面中只露出上半身，背对着镜头，她面前站着一个穿着军绿色上衣的男士[0.698, 0.604, 0.786, 0.999]。她似乎正朝着大门走去。根据穿着和行为可推断她可能是一名顾客。\n\n[0.69, 0.235, 0.769, 0.618] 这是一个穿着紫色带有图案的长袖上衣、蓝色长牛仔裤和黑色运动鞋的男士。他的前面停放着一辆黑色汽车，他正在观察这辆车。根据穿着和行为可推断他是一名顾客。\n\n[0.698, 0.604, 0.786, 0.999] 这是一个穿着军绿色羽绒服、戴着浅蓝色口罩的男子。他的左手边是一个穿着黑色衣服的女士[0.657, 0.766, 0.775, 0.997]，她们似乎正一起走向该店的大门。根据穿着和行为可推断他可能是一名顾客。\n\n[0.789, 0.474, 0.865, 0.976] 这是一个穿着银色长袖羽绒服、浅咖色长裤的女士。她头戴黄色帽子、浅蓝色口罩和一副眼镜。她的双手缩在袖子里，正朝着画面的右侧站立。她可能在与她面前穿着蓝色羽绒服的男士[0.923, 0.492, 0.998, 0.967]说话。根据穿着和行为可推断她可能是一名顾客。\n\n[0.9, 0.726, 0.971, 0.999] 这是一个穿着红色的长袖羽绒服、深色长裤和头戴白色口罩的小男孩。左边穿蓝色长袖羽绒服的男士[0.923, 0.492, 0.998, 0.967]似乎正牵着他的左手。根据年龄可推断他可能是一名顾客。\n\n[0.923, 0.492, 0.998, 0.967] 这是一个穿着蓝色长袖羽绒服、戴着浅蓝色口罩的男士。他的腰部挎着一个水壶，正回头向左看向他身后一个穿着银色羽绒服的女士[0.789, 0.474, 0.865, 0.976]。根据行为可推断他、女士[0.789, 0.474, 0.865, 0.976]和小男孩[0.9, 0.726, 0.971, 0.999] 是同一个组。同时也可推断他可能是一名顾客。"
    """
    bboxes = []
    for line in caption.split('\n'):
        if line == '':
            continue
        # 提取出每行开头的坐标，例如：[0.789, 0.474, 0.865, 0.976]
        line = line.strip()
        if line[0] != '[':
            continue

        end = line.find(']')
        if end == -1:
            continue
        line = line[1:end]
        line = line.replace(' ', '')
        line = line.split(',')
        line = list(map(float, line))
        bboxes.append(line)
    return bboxes


def parse_bbox_caption_pairs_from_label(label):
    """
    label: str
    return: list of bboxes
    "[0.506, 0.187, 0.571, 0.531] 这是一个穿着黑色长袖外套、黑色长裤和黑色休闲鞋的男士。他戴着白色口罩，左边紧靠着反光墙，右边是一位女士[0.542, 0.209, 0.625, 0.548]，他们站在展台前，正看着前方的黑色展车。可推断他和女士[0.542, 0.209, 0.625, 0.548]是一个组。根据穿着和行为可推断他可能是一名顾客。\n\n[0.542, 0.209, 0.625, 0.548] 这是一个穿着黄绿相间并带有卡通图案的长袖上衣、白色短裙和白色休闲鞋的女士。她左手挽着一位男士[0.542, 0.209, 0.625, 0.548]，他们站在展台前，正看着前方的黑色展车。可推断她和男士[0.506, 0.187, 0.571, 0.531]是一个组。根据穿着和行为可推断她可能是一名顾客。\n\n[0.621, 0.377, 0.71, 0.874] 这是一个穿着黑白色的长袖羽绒服、黑色长裤和深色休闲鞋的男士。他手里拿着一个手机，正站在画面中央看手机。根据穿着和行为可推断他可能是一名顾客。\n\n[0.657, 0.766, 0.775, 0.997] 这是一个穿着黑色上衣的女士，她留着长发，在画面中只露出上半身，背对着镜头，她面前站着一个穿着军绿色上衣的男士[0.698, 0.604, 0.786, 0.999]。她似乎正朝着大门走去。根据穿着和行为可推断她可能是一名顾客。\n\n[0.69, 0.235, 0.769, 0.618] 这是一个穿着紫色带有图案的长袖上衣、蓝色长牛仔裤和黑色运动鞋的男士。他的前面停放着一辆黑色汽车，他正在观察这辆车。根据穿着和行为可推断他是一名顾客。\n\n[0.698, 0.604, 0.786, 0.999] 这是一个穿着军绿色羽绒服、戴着浅蓝色口罩的男子。他的左手边是一个穿着黑色衣服的女士[0.657, 0.766, 0.775, 0.997]，她们似乎正一起走向该店的大门。根据穿着和行为可推断他可能是一名顾客。\n\n[0.789, 0.474, 0.865, 0.976] 这是一个穿着银色长袖羽绒服、浅咖色长裤的女士。她头戴黄色帽子、浅蓝色口罩和一副眼镜。她的双手缩在袖子里，正朝着画面的右侧站立。她可能在与她面前穿着蓝色羽绒服的男士[0.923, 0.492, 0.998, 0.967]说话。根据穿着和行为可推断她可能是一名顾客。\n\n[0.9, 0.726, 0.971, 0.999] 这是一个穿着红色的长袖羽绒服、深色长裤和头戴白色口罩的小男孩。左边穿蓝色长袖羽绒服的男士[0.923, 0.492, 0.998, 0.967]似乎正牵着他的左手。根据年龄可推断他可能是一名顾客。\n\n[0.923, 0.492, 0.998, 0.967] 这是一个穿着蓝色长袖羽绒服、戴着浅蓝色口罩的男士。他的腰部挎着一个水壶，正回头向左看向他身后一个穿着银色羽绒服的女士[0.789, 0.474, 0.865, 0.976]。根据行为可推断他、女士[0.789, 0.474, 0.865, 0.976]和小男孩[0.9, 0.726, 0.971, 0.999] 是同一个组。同时也可推断他可能是一名顾客。"
    """
    bbox_caption_pairs = []
    for line in label.split('\n'):
        if line == '':
            continue
        # 提取出每行开头的坐标，例如：[0.789, 0.474, 0.865, 0.976]
        line = line.strip()
        if line[0] != '[':
            continue

        end = line.find(']')
        if end == -1:
            continue

        bbox = line[1:end]
        caption = line[end+1:].strip()
        bbox = bbox.replace(' ', '')
        bbox = bbox.split(',')
        bbox = list(map(float, bbox))
        
        if caption == '':
            continue

        bbox_caption_pairs.append((bbox, caption))
    return bbox_caption_pairs


def plot_boxes_on_image_resave(img_path, boxes, save_path):
    """ Plot boxes on image and save it.
    Args:
        img_path: path of image
        boxes: boxes to be plotted
    Returns:
        None
    """
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(img_path)
    boxes = denorm(boxes, img.shape[0], img.shape[1])
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    image_id = img_path.split('images/')[-1]
    img_path = Path(save_path) / image_id
    if not Path(img_path).parent.exists():
        Path(img_path).parent.mkdir(exist_ok=True, parents=True)
    
    # If this image has exist -> skip save it by overwrite
    if not Path(img_path).exists():
        print(f'saving image: {img_path}')
        cv2.imwrite(img_path.as_posix(), img)
        return
    
    print(f'image {img_path} already exist, skip saving')
    return


def parse_label_result_v1(label_path, img_list_path):
    """ Parse label of v1 into v2 format
    """
    img_list = json.loads(open(img_list_path, 'r').read())
    img_list.sort()

    labels = json.loads(open(label_path, 'r').read())

    labels_tov2 = {}
    for i, (index, label) in enumerate(labels.items()):
        idx = int(index)
        img_dir = img_list[idx]

        labels_tov2[i] = {
            "label": label,
            "global_caption": "",
            "img": img_dir
        }
    return labels_tov2


def merge_labels(labels_v1, labels_v2):
    if not USE_EN:
        labels = labels_v2
    else:
        labels = {}
        for i, (idx, label) in enumerate(labels_v2.items()):
            if int(idx) > Filter_NUM:
                continue
            try:
                label['label'] = label['caption']  # caption is en caption, label is ch caption
                labels[str(i)] = label
            except:
                print(label)
                import pdb; pdb.set_trace()
    for i, (idx, label) in enumerate(labels_v1.items()):
        # 这里之所以使用 -idx 表示id，是为了不改变label_v2的id顺序，因为后面需要根据id过滤掉还没进行check的样本
        labels[str(-(idx + 1))] = label
    return labels


###################################### 以box为单位进行描述 ###############################
# 每个person单独拿出来作为一个样本
def convert_label_to_lavis_format(label_path, new_label_path, images_save_path, label_path_v1=None, train_img_list_v1=None, is_test=False):
    with open(label_path, 'r') as f:
        labels = json.load(f)

    if label_path_v1 is not None:
        labels_v1 = parse_label_result_v1(label_path_v1, train_img_list_v1)
        # merge labels_v1 and labels
        labels = merge_labels(labels_v1, labels)

        with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
    
    # Parse original label and convert to llava: key: id,image,conversations
    labels_lavis = []
    num_boxes_all = 0

    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    
    idx_bbox = 0
    for idx, label in tqdm(labels.items()):
        if is_test:
            if int(idx) < Filter_NUM or int(idx) > MaxNum_forEval:
                continue
        else:
            if int(idx) > Filter_NUM:
                continue

        img = label['img']  # "img": "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01002_20210617125000/3204.jpg"
        image_id = img.split('images/')[-1]
        caption = label['caption'] if 'caption' in label else label['label']
        global_caption = label['global_caption']

        bbox_caption_pairs = parse_bbox_caption_pairs_from_label(caption)
        num_boxes_all += len(bbox_caption_pairs)
        bbox_caption_pairs = sorted(bbox_caption_pairs, key=lambda x: x[0])

        boxes_str = ''
        for i, (box, caption) in enumerate(bbox_caption_pairs):
            boxes_str += f'{i}: ' + str(box) + '\n'

            idx_bbox += 1
            sample = {
                "image": image_id,
                "image_id": idx_bbox,
                "bbox": box,
                "dense_caption": f"{caption}<bbox>{box}"
            }

            labels_lavis.append(sample)
    
    # Save label result
    with open(new_label_path, 'w', encoding='utf-8') as f:
        json.dump(labels_lavis, f, ensure_ascii=False, indent=4)
    
    print(f'done! Saving into: {new_label_path}')
    print(f"Number of all of boxes: {num_boxes_all}")


if __name__ == "__main__":
    GEN_TEST_DATA = True
    if USE_EN:
        label_path_v1 = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_en.json"
        label_path = '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json'
        new_label_path = '/training/wphu/Dataset/llava/AibeeQA/label_result_530_llava-style-box_caption_en.json' if not GEN_TEST_DATA else '/training/wphu/Dataset/llava/AibeeQA/label_result_530_llava-style-box_caption_en_test.json'
        prompt_text_path = './data/prompt_text_en.txt'
    else:
        label_path_v1 = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1.json"
        label_path = '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2.json'
        new_label_path = '/training/wphu/Dataset/llava/AibeeQA/label_result_530_llava-style-box_caption.json' if not GEN_TEST_DATA else '/training/wphu/Dataset/llava/AibeeQA/label_result_530_llava-style-box_caption_test.json'
        prompt_text_path = './data/prompt_text.txt'
    
    train_img_list_v1 = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json"

    # label_path = './data/label_result_v2_400.json'
    
    # new_label_path = './data/label_result_v2_400_llava.json'

    images_save_path = '/training/wphu/Dataset/llava/AibeeQA/images'

    Instructions = Path(prompt_text_path).read_text()  # TODO: extra coords of all bboxes

    # convert_label_to_lavis_format(label_path, new_label_path, images_save_path, label_path_v1, train_img_list_v1)

    # Generate test.json only use label_result_v2_400
    convert_label_to_lavis_format(label_path, new_label_path, images_save_path, is_test=True)


