import streamlit as st
import json
from PIL import Image, ImageDraw
import io
import streamlit.components.v1 as components

# 设置页面配置为 "wide"，以占据整个屏幕宽度
# st.set_page_config(layout="wide")

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

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_width, image_height = image.size

    # 获取图像的绘制上下文
    draw = ImageDraw.Draw(image)

    # 将JavaScript代码提取为单独的.js文件，例如 "coordinates.js"
    js_code = open("annotation_tools/coordinates.js", "r").read()

    # 在Streamlit界面中注入JavaScript代码
    # st.markdown(js_code, unsafe_allow_html=True)
    components.html(js_code, height=600)
    st.markdown("injecting js ...", unsafe_allow_html=True)
    # components.iframe("http://www.bing.com/", height=600, width=1000)

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
