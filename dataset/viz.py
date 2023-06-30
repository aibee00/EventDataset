import gradio as gr
import json
import cv2
import os
import os.path as osp

class ImageVisualizer:
    def __init__(self, dataset_path, dataset_creater):
        self.label_path = dataset_creater.label_path
        self.dataset_path = dataset_path
        self.prompt_descriptor = dataset_creater.prompt_descriptor
        self.img_annos = None
        self.current_index = 0

    def draw_img_prompt(self, img, annotation):
        # 1. 将annotation解意为prompt
        prompt = self.prompt_descriptor.parse_annotation(annotation)
        # 2. 绘制prompt
        img = self.prompt_descriptor.draw_prompt(img, prompt)
        return img

    def load_data(self):
        """ label_path 内容：
        {
            "ch01001_20210717201840.jpg": {
                "img": "/ssd/wphu/Dataset/GACNE-guangzhou-xhthwk-20210717/imgs/ch01001_20210717201840.jpg",
                "annotation": "顾客:(['p-1969', 'p-245', 'p-1018', 'p-1953'])是同一批次;顾客p-1018正进入该店大门"
            },
            "ch01001_20210717201842.jpg": {
                "img": "/ssd/wphu/Dataset/GACNE-guangzhou-xhthwk-20210717/imgs/ch01001_20210717201842.jpg",
                "annotation": "顾客:(['p-1969', 'p-245', 'p-1018', 'p-1953'])是同一批次;顾客p-1018正进入该店大门"
            },
            "ch01001_20210717201921.jpg": {
                "img": "/ssd/wphu/Dataset/GACNE-guangzhou-xhthwk-20210717/imgs/ch01001_20210717201921.jpg",
                "annotation": "顾客p-1018正走出该店大门"
            },
        }
        """
        with open(self.label_path, "r", encoding='utf-8') as f:
            self.img_annos = json.load(f)

    def get_current_image(self):
        if self.img_annos is None:
            self.load_data()
        img_name, annotation = list(self.img_annos.items())[self.current_index]
        img_path = annotation["img"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # img = self.draw_img_prompt(img, annotation)
        return img

    def get_image_by_index(self, index):
        if self.img_annos is None:
            self.load_data()
        img_name, annotation = list(self.img_annos.items())[index]
        img_path = annotation["img"]
        img = cv2.imread(img_path)
        # img = self.draw_img_prompt(img, annotation)
        return img

    def save_current_image(self):
        if self.img_annos is None:
            self.load_data()
        img_name, annotation = list(self.img_annos.items())[self.current_index]
        img_path = annotation["img"]
        img = cv2.imread(img_path)
        img = self.draw_img_prompt(img, annotation)
        img_dir = osp.join(self.dataset_path, "viz_img_prompt")
        if not osp.exists(img_dir):
            os.makedirs(img_dir)

        img_path = osp.join(img_dir, img_name)
        cv2.imwrite(img_path, img)
        print(f"Img saved to: {img_path}")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1

    def prev_annotation(self):
        if self.current_index > 0:
            self.current_index -= 1

    def next_image(self):
        if self.current_index < len(self.img_annos) - 1:
            self.current_index += 1

    def next_annotation(self):
        if self.current_index < len(self.img_annos) - 1:
            self.current_index += 1

def viz_img_prompt(dataset_path, dataset_creater):
    visualizer = ImageVisualizer(dataset_path, dataset_creater)

    def get_current_image():
        return visualizer.get_current_image()
    
    def get_current_annotation():
        return visualizer.img_annos[list(visualizer.img_annos.keys())[visualizer.current_index]]["annotation"]
    
    def get_current_context():
        return visualizer.img_annos[list(visualizer.img_annos.keys())[visualizer.current_index]]["context"]

    def prev_image():
        visualizer.prev_image()
        return get_current_image(), get_current_annotation(), get_current_context(), f"current index: {visualizer.current_index}"

    def next_image():
        visualizer.next_image()
        return get_current_image(), get_current_annotation(), get_current_context(), f"current index: {visualizer.current_index}"

    def save_current_image():
        visualizer.save_current_image()


    with gr.Blocks() as demo:
        with gr.Column():    # 列排列
            image = gr.Image(value=get_current_image())
        
        with gr.Column():    # 列排列
            annotation = gr.Textbox(value=get_current_annotation())

        with gr.Column():    # 列排列
            context = gr.Textbox(value=get_current_context())
        
        with gr.Row():       # 行排列
            index = gr.Textbox(value=f"current index: {visualizer.current_index}")
            prev_btn = gr.Button("prev")
            next_btn = gr.Button("next")
            save_btn = gr.Button("save")

        next_btn.click(next_image, inputs=[], outputs=[image, annotation, context, index])
        prev_btn.click(prev_image, inputs=[], outputs=[image, annotation, context, index])
        save_btn.click(save_current_image)

    demo.launch(share=True)


