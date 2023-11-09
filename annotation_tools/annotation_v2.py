""" V2 说明

运行此脚本请用：
streamlit run annotation_v2.py img_list_path label_path

该版本主要是针对gpt4v标注而建立，主要不同如下：
- 更改了保存格式，原来保存是一个字典，key是index，value是caption，现在改为value是一个字典,包括image_name和caption
- 标注顺序是乱序的，原来是先sorted的。
- 数据多了，由原来的1000改为4000，原来数据只有hongqi和vw_hefei的，现在是5店天的数据百万张图中随机选取4000张
"""
import streamlit as st
import json
import sys
from pathlib import Path
import cv2
import os 

from utils import denorm, get_label_info, plot_bboxes_on_image, H, W
from copy import deepcopy

VERSION="v2"
MANUAL_LABEL = False  # 人工标注: True, GPT-4v标注: False, 人工标注图片顺序是sorted的对应的index不能改变否则就乱了

# 设置页面配置为 "wide"，以占据整个屏幕宽度
st.set_page_config(layout="wide")

# Load image paths from a JSON file
# img_list_path = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json"  # 从最开始的数据可能只包含红旗和合肥的
img_list_path = f"/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list_{VERSION}.json"  # 数据增强后的从1138714张图片中随机选的
# label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_onlyperson_train.json"
# label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_train.json"
label_path = "/training/wphu/Dataset/lavis/eventgpt/gpt4v_annotaions/weiding_detections_4000.json"

if len(sys.argv) > 1:
    img_list_path = sys.argv[1]

if len(sys.argv) > 2:
    label_path = sys.argv[2]

root = Path(img_list_path).parent
# save_path = root / "label_result.json"
save_path = root / f"label_result_{VERSION}.json"

result = json.loads(open(img_list_path, 'r').read())
if MANUAL_LABEL:
    result.sort()

labels = json.loads(open(label_path, 'r').read())
labels.sort(key=lambda x: x['image'])

# convert to dict, key is image_id, value is label_infos
label_map = {label['image']: label for label in labels}

# Initialize global variables
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

# load from label_result.json
if not save_path.exists():
    # mkdir
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("{}")

label_result = json.loads(open(save_path, 'r').read())
st.session_state.annotations = label_result

if not getattr(st.session_state, 'current_index', None):
    setattr(st.session_state, 'current_index', 0)

current_index = st.session_state.current_index

# Initialize data dictionary
data_dict = {}

global input_bar

def get_image_name_for_gpt4v(img_path):
    # 获取子文件夹名称
    subfolder_name = img_path.split("/")[7]

    ch_name = img_path.split("/")[-2]
    
    # 从路径中获取图片文件名
    img_file_name = os.path.basename(img_path)
    
    # 在文件名前添加子文件夹名称作为前缀
    new_img_file_name = f"{subfolder_name}__{ch_name}__{img_file_name}"
    return new_img_file_name

# Function to load and display the current image and caption
def load_current_image():
    if 0 <= current_index < len(result):
        image_path = result[current_index]

        # Get bounding box info
        bboxes_norm = get_label_info(image_path, label_map, "bbox")
        bboxes_norm.sort(key=lambda x: x[0])  # Sort bboxes by x coordinate
        bboxes = denorm(bboxes_norm, H, W)
        img = plot_bboxes_on_image(image_path, bboxes)

        data_dict["img"] = image_path

        if str(current_index) in st.session_state.annotations:
            data_dict["label"] = st.session_state.annotations[str(current_index)].get("label", "")  # Get existing caption
            data_dict["global_caption"] = st.session_state.annotations[str(current_index)].get("global_caption", "")  # Get existing caption
        else:
            data_dict["label"] = ""
            data_dict["global_caption"] = ""
        
        st.write(f"第{current_index}/{len(result)}张图片: {image_path}")
        st.markdown(f"Image name for gpt4v: {get_image_name_for_gpt4v(image_path)}")
        # st.image(data_dict["img"])
        # 将 OpenCV 图像从 BGR 格式转换为 RGB 格式
        opencv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(opencv_image_rgb)
        with col2:
            # st.write(f"bboxes:")
            bboxes_viz = {}
            bbox_str = ""
            for i, bbox in enumerate(bboxes_norm):
                bboxes_viz[i] = bbox
                # st.write(f"{i}: {bbox}")
                bbox_str += f"{i}: {bbox}\n"
            
            if bbox_str == "":
                bbox_str = "没有找到任何bbox"
            
            st.text_area("bboxes:", key=f"bbox_str_{current_index}", value=f"{bbox_str}", height=400)

            # 记录全局的描述，包含推理部分
            data_dict["global_caption"] = \
                st.text_area("全局描述或者推理相关内容:", key=f"global_desc_{current_index}", value=f"{data_dict['global_caption']}", height=600)
        
        label_key = f"label_input_{current_index}"  # Generate unique key
        data_dict["label"] = st.text_area("标签", key=label_key, value=data_dict["label"], height=400)  # Display and edit caption

# Function to save the caption
def save_caption():
    caption_key = f"label_input_{current_index}"
    data_dict["label"] = st.session_state.get(caption_key, "")  # Get caption from the input box

    global_caption_key = f"global_desc_{current_index}"
    data_dict["global_caption"] = st.session_state.get(global_caption_key, "")  # Get caption from the input box
    data_dict["img"] = result[current_index]
    st.session_state.annotations[str(current_index)] = deepcopy(data_dict)
    with col2:
        st.write(f"第{current_index}/{len(result)}张图片的标签已保存。")

# Streamlit app title
st.title("图像标注")

# Create a column for image display and next/previous buttons
col1, _, col2, col3 = st.columns([30, 1, 9, 16])

with col3:
    if st.button("⬆️") and current_index > 0:
        save_caption()
        current_index -= 1
        st.session_state.current_index = current_index

with col3:
    if st.button("⬇️") and current_index < len(result) - 1:
        save_caption()
        current_index += 1
        st.session_state.current_index = current_index

with col1:
    load_current_image()

with col3:
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns([1, 1, 1, 2])
    with sub_col1:
        # Create a save button to save the caption
        if st.button("保存"):
            save_caption()
    # Display all saved captions
    st.write("已保存的标签:")
    st.write({str(current_index): st.session_state.annotations.get(str(current_index), {})}, height=800)
    # st.text_area("已保存的标签:", key=f"label_saved_{current_index}", value=st.session_state.annotations, height=800)

    with sub_col2:
        if st.button("备份"):
            # Save all results to a JSON file
            # sort annotations by key
            annotations = {int(k): v for k, v in st.session_state.annotations.items()}
            annotations = {int(k): v for k, v in sorted(annotations.items()) if v and v['label']}
            # with open('./data/label_result_bak.json', 'w', encoding="utf-8") as f:
            with open(f'./data/label_result_{VERSION}_bak.json', 'w', encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)

    with sub_col3:
        if st.button("跳转到") and current_index < len(result) - 1:
            save_caption()
            st.session_state.current_index = current_index

    with sub_col4:
        current_index = st.number_input("Index", value=int(current_index), min_value=0, max_value=len(result) - 1)
        if current_index is not None:
            st.session_state.current_index = current_index

# Save all results to a JSON file
with open(save_path, 'w', encoding="utf-8") as f:
    annotations = {int(k): v for k, v in st.session_state.annotations.items()}
    annotations = {int(k): v for k, v in sorted(annotations.items()) if v and v['label']}
    json.dump(annotations, f, ensure_ascii=False, indent=2)
