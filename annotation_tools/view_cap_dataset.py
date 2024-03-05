import gradio as gr
import json
import os

activity_name = "person_stands_up_from_floor"
label_path = "/training/wphu/Dataset/lavis/from_cap/annotations/dense_captions/"
image_dir = "/training/wphu/Dataset/lavis/from_cap/images/"

# 加载label.json中的数据
label_file = os.path.join(label_path, activity_name, "dense_captions_of_cap.json")
with open(label_file, "r") as file:
    labels = json.load(file)

# 将字典的键（图片文件名）转换成列表以方便索引
images = list(labels.keys())
# images = [os.path.join(image_dir, activity_name, image) for image in images]
current_index = [0]  # 使用列表来保存当前索引，因为需要在函数内修改

def update_image(direction):
    """
    根据用户点击的按钮更新图片和caption。
    
    :param direction: 方向，可以是'next'或'prev'。
    :return: 返回图片路径和对应的caption。
    """
    if direction == "next":
        current_index[0] = (current_index[0] + 1) % len(images)
    elif direction == "prev":
        current_index[0] = (current_index[0] - 1) % len(images)
    
    image = images[current_index[0]]
    caption = labels[image].strip()
    image = os.path.join(image_dir, activity_name, image)  # 添加图片路径前缀，以便gradio能够加载图片文件。
    return image, caption

# 创建Gradio界面
iface = gr.Interface(
    fn=update_image,
    inputs=gr.Radio(["prev", "next"], label="浏览图片"),
    outputs=[gr.Image(type="filepath"), gr.Textbox(label="Caption")],
    title="图片浏览器",
    description="查看图片及其caption。点击'下一个'或'上一个'按钮来浏览。"
)

# 启动Gradio应用
iface.launch()
