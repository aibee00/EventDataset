import streamlit as st
import json
from PIL import Image, ImageDraw
import io
import streamlit.components.v1 as components

st.title("Bounding Box Annotation Tool")

# 初始化标注数据
annotations = []

# 用于保存标注数据的函数
def save_annotations(annotations):
    with open("annotations.json", "w") as f:
        json.dump(annotations, f)

# Streamlit界面
uploaded_image = st.file_uploader("上传图像", type=["jpg", "png", "jpeg"])
canvas = st.empty()
clear_button = st.button("清除标注")
save_button = st.button("保存标注")
coordinates_element = st.empty()

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_width, image_height = image.size

    # 获取图像的绘制上下文
    draw = ImageDraw.Draw(image)

    # 在Streamlit界面中注入JavaScript代码
    js_code = open("annotation_tools/annotate.js", "r").read()
    components.html(js_code, height=0)

    # 显示图像
    canvas.image(image, use_column_width=True)

# 清除标注
if clear_button:
    annotations = []
    image = Image.open(uploaded_image)
    draw = ImageDraw.Draw(image)
    canvas.image(image, use_column_width=True)

# 保存标注
if save_button:
    save_annotations(annotations)
    st.success("标注已保存")

# 显示坐标
if annotations:
    coordinates_element.json(annotations)
