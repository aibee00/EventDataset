import streamlit as st
import cv2
import numpy as np

# 图像路径
image_path = "/ssd/wphu/Dataset/lavis/eventgpt/debug/img_1.jpg"

# 显示图像
st.image(image_path, use_column_width=True)

# JavaScript 代码，用于在图像上获取鼠标坐标
javascript = """
<script>
let coordinates = [];

document.querySelector("img").addEventListener("click", function(event) {
    const x = event.offsetX;
    const y = event.offsetY;
    coordinates.push([x, y]);
    console.log("Clicked at:", x, y);
    // 将坐标数据发送到 Streamlit 应用
    Streamlit.setComponentValue(coordinates);
});

document.querySelector("#show-coordinates").addEventListener("click", function() {
    console.log("Coordinates:", coordinates);
    alert("Coordinates: " + JSON.stringify(coordinates));
});
</script>
"""

# 将 JavaScript 代码添加到页面
st.markdown(javascript, unsafe_allow_html=True)

# 添加一个按钮，用于显示已获取的坐标
if st.button("显示坐标", key="show-coordinates"):
    if "coordinates" in st.session_state:
        coordinates = st.session_state.coordinates
        if coordinates:
            # 加载图像并绘制 bounding box
            image = cv2.imread(image_path)
            for (x, y) in coordinates:
                cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
            
            # 将 OpenCV 图像从 BGR 格式转换为 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 显示带有 bounding box 的图像
            st.image(image_rgb, use_column_width=True)
        else:
            st.write("请先单击图像获取坐标点。")
    else:
        st.write("请先单击图像获取坐标点。")
