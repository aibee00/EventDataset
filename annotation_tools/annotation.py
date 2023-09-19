import streamlit as st
import json
from copy import deepcopy

print("restart")

result = json.loads(open('data/dataset_img_list.json', 'r').read())
result.sort()

current_image_index = st.session_state.get("counter", 0)

# 初始化字典和图片路径列表
annotations =  st.session_state.get("annotations", {})
data_dict = {}
image_paths = result #["image1.jpg", "image2.jpg", "image3.jpg"]  # 以列表形式提供图片路径


def load_first_image():
    global current_image_index
    if int(current_image_index) < len(image_paths):
        image_path = image_paths[current_image_index]
        data_dict["img"] = image_path
        data_dict["label"] = ""  # 清空标注结果
        st.write("第{}/{}张: {}".format(
            current_image_index, 
            len(result),
            image_paths[current_image_index]))
    else:
        st.write("No more images.")

# 点击"下一个"按钮加载下一张图片
def load_next_image():
    global current_image_index
    if int(current_image_index) < len(image_paths):
        image_path = image_paths[current_image_index]
        data_dict["img"] = image_path
        data_dict["label"] = ""  # 清空标注结果
        current_image_index += 1
    else:
        st.write("No more images.")

# 点击"上一个"按钮加载上一张图片
def load_pre_image():
    global current_image_index
    if int(current_image_index) >= 0 and int(current_image_index) < len(image_paths):
        image_path = image_paths[current_image_index]
        data_dict["img"] = image_path
        data_dict["label"] = ""  # 清空标注结果
        current_image_index -= 1
    else:
        st.write("No more images.")

# 点击"下一个"或者"上一个"按钮保存标注结果
def pre_next_save_label():
    # global current_image_index
    print(current_image_index)
    if data_dict["img"] not in annotations:
        data_dict["label"] = st.session_state.label_input
        annotations[data_dict["img"]] = deepcopy(data_dict)  # 将当前标注结果添加到列表中
    st.write("第{}/{}张: {}".format(
            current_image_index, 
            len(result),
            image_paths[current_image_index]))

# 点击"Submit"按钮保存标注结果
def save_label():
    # global current_image_index
    data_dict["label"] = st.session_state.label_input
    print(current_image_index)
    annotations[data_dict["img"]] = deepcopy(data_dict)  # 将当前标注结果添加到列表中
    st.write("第{}/{}张: {}".format(
            current_image_index, 
            len(result),
            image_paths[current_image_index]))

st.title("Image Annotation")

# 加载第一张图片
load_first_image()

# 显示当前图片和标注输入框
st.image(data_dict["img"])
label_input = st.text_area("Label", key="label_input")

col1, col2 = st.columns(2)

with col1:
    # 当点击"Next"按钮时，加载下一张图片并保存标注结果
    if st.button("下一个"):
        load_next_image()
        pre_next_save_label()
        st.session_state["counter"] = current_image_index

with col2:
    # 当点击"Next"按钮时，加载下一张图片并保存标注结果
    if st.button("上一个"):
        load_pre_image()
        pre_next_save_label()
        st.session_state["counter"] = current_image_index

st.columns(1)

# 当点击"Submit"按钮时，仅保存标注结果
if st.button("Submit"):
    save_label()

# 显示所有标注结果
st.session_state["annotations"] = annotations
st.write("Annotations:")
st.write(annotations)

# 保存所有结果
with open("data/label_result.json", 'w', encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)
