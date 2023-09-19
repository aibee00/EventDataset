import streamlit as st
import json
import sys

# 设置页面配置为 "wide"，以占据整个屏幕宽度
st.set_page_config(layout="wide")

# Load image paths from a JSON file
label_path = "data/dataset_img_list.json"

if len(sys.argv) > 1:
    label_path = sys.argv[1]

result = json.loads(open(label_path, 'r').read())
result.sort()

# Initialize global variables
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

current_index = st.session_state.current_index

# Initialize data dictionary
data_dict = {}

# Function to load and display the current image and caption
def load_current_image():
    if 0 <= current_index < len(result):
        image_path = result[current_index]
        data_dict["img"] = image_path
        data_dict["label"] = st.session_state.annotations.get(str(current_index), "")  # Get existing caption
        st.write(f"第{current_index}/{len(result)}张图片: {image_path}")
        st.image(data_dict["img"])
        label_key = f"label_input_{current_index}"  # Generate unique key
        data_dict["label"] = st.text_area("标签", key=label_key, value=data_dict["label"])  # Display and edit caption

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
col1, col2, col3 = st.columns([15, 1, 10])

# with col1:
#     # Display the current image and caption
#     load_current_image()

with col3:
    if st.button("上一个") and current_index > 0:
        save_caption()
        current_index -= 1
        st.session_state.current_index = current_index
        with col1:
            # Display the current image and caption
            load_current_image()

with col3:
    if st.button("下一个") and current_index < len(result) - 1:
        save_caption()
        current_index += 1
        st.session_state.current_index = current_index
        with col1:
            # Display the current image and caption
            load_current_image()

with col3:
    # Create a save button to save the caption
    if st.button("保存"):
        save_caption()
    # Display all saved captions
    st.write("已保存的标签:")
    st.write(st.session_state.annotations)




# Save all results to a JSON file
with open("data/label_result.json", 'w', encoding="utf-8") as f:
    json.dump(st.session_state.annotations, f, ensure_ascii=False, indent=2)
