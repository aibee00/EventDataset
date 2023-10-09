import streamlit as st
import json
import sys
from pathlib import Path
import cv2

from utils import denorm, get_label_info, plot_bboxes_on_image, H, W

# 设置页面配置为 "wide"，以占据整个屏幕宽度
st.set_page_config(layout="wide")

# Load image paths from a JSON file
img_list_path = "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json"
label_path = "/training/wphu/Dataset/lavis/eventgpt/annotations/vqa_oracle_onlyperson_train.json"

if len(sys.argv) > 1:
    img_list_path = sys.argv[1]

if len(sys.argv) > 2:
    label_path = sys.argv[2]

root = Path(img_list_path).parent
save_path = root / "label_result.json"

result = json.loads(open(img_list_path, 'r').read())
result.sort()

labels = json.loads(open(label_path, 'r').read())
labels.sort(key=lambda x: x['image'])

# convert to dict, key is image_id, value is label_infos
label_map = {label['image']: label for label in labels}

# Initialize global variables
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

# load from label_result.json
label_result = json.loads(open(save_path, 'r').read())
st.session_state.annotations = label_result

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

current_index = st.session_state.current_index

# Initialize data dictionary
data_dict = {}

global input_bar

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
        data_dict["label"] = st.session_state.annotations.get(str(current_index), "")  # Get existing caption
        st.write(f"第{current_index}/{len(result)}张图片: {image_path}")
        # st.image(data_dict["img"])
        # 将 OpenCV 图像从 BGR 格式转换为 RGB 格式
        opencv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(opencv_image_rgb)
        with col2:
            st.write(f"bboxes:")
            bboxes_viz = {}
            for i, bbox in enumerate(bboxes_norm):
                bboxes_viz[i] = bbox
                st.write(f"{i}: {bbox}")
        
        label_key = f"label_input_{current_index}"  # Generate unique key
        data_dict["label"] = st.text_area("标签", key=label_key, value=data_dict["label"], height=400)  # Display and edit caption

# Function to save the caption
def save_caption():
    caption_key = f"label_input_{current_index}"
    data_dict["label"] = st.session_state.get(caption_key, "")  # Get caption from the input box
    st.session_state.annotations[str(current_index)] = data_dict["label"]
    with col2:
        st.write(f"第{current_index}/{len(result)}张图片的标签已保存。")

# Streamlit app title
st.title("图像标注")

# Create a column for image display and next/previous buttons
col1, _, col2, col3 = st.columns([15, 1, 4, 8])

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
    st.write({str(current_index): st.session_state.annotations.get(str(current_index), "空")}, height=800)
    # st.text_area("已保存的标签:", key=f"label_saved_{current_index}", value=st.session_state.annotations, height=800)

    with sub_col2:
        if st.button("备份"):
            # Save all results to a JSON file
            with open('./data/label_result_bak.json', 'w', encoding="utf-8") as f:
                json.dump(st.session_state.annotations, f, ensure_ascii=False, indent=2)

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
    json.dump(st.session_state.annotations, f, ensure_ascii=False, indent=2)
