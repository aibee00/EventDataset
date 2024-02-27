"""
我们抽帧后的数据只有动作的短语，我们需要将其转为详细的描述。
这里我们选择使用现成的开源模型帮我们实现。
例如，我们可以使用 video_caption 模型 或者 image_caption 模型。
video_caption 模型：
- video_llava，vid2seq等
- llava, blip2等
"""

import os
import json
import argparse


from tqdm import tqdm
from pathlib import Path

from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

from abc import ABC, abstractmethod


LLAVA_CHECKPOINT_PATH = "/training/wphu/Checkpoints/llava/llava-1.5-7b-hf"


# 抽象类
class VideoCaptionModel(ABC):
    @abstractmethod
    def get_caption(self, video_path, activity_name, max_length=256):
        raise NotImplementedError


class LlavaModel(VideoCaptionModel):
    def __init__(self):
        self.device = "cuda"
        self.model = LlavaForConditionalGeneration.from_pretrained(LLAVA_CHECKPOINT_PATH)
        # 将 model 量化
        self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(LLAVA_CHECKPOINT_PATH)
        self.prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"

    def update_prompt(self, activity_name):
        self.prompt = f"<image>\nUSER: What's the content of the image? Note: The activity is '{activity_name.replace('_',' ')}, Please Describe the actions in the figure in more detail based on the given action phrase, focusing on fine-grained action details'\nASSISTANT:"

    def get_caption(self, img_path, activity_name, max_length=256):
        self.update_prompt(activity_name)
        image = Image.open(img_path)  
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=max_length)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output


class GenDenseCaptionForClips(object):
    def __init__(self, cap_dir):
        self.cap_dir = cap_dir
        self.annotation_dir = os.path.join(cap_dir, "annotations")
        self.image_dir = os.path.join(cap_dir, "images")

        self.dense_caption_dir = Path(self.annotation_dir)
        self.dense_caption_dir.mkdir(parents=True, exist_ok=True)
        self.dense_caption_file = os.path.join(self.dense_caption_dir, "dense_captions_of_cap.json")

        self.image_caption_models = {}  # registry of caption models

        self.save_interval = 20  # 每隔10个图片保存一次


    def register_model(self, model_name, model):
        self.image_caption_models[model_name] = model

    def save_result(self, dense_captions):
        with open(self.dense_caption_file, "w", encoding='utf-8') as f:
            json.dump(dense_captions, f, indent=4, ensure_ascii=False)
        print(f"Save dense captions to {self.dense_caption_file}")

    def load_result(self):
        if os.path.exists(self.dense_caption_file):
            with open(self.dense_caption_file, "r", encoding='utf-8') as f:
                dense_captions = json.load(f)
            print(f"Load dense captions from {self.dense_caption_file}")
            return dense_captions
        return {}

    def gen_dense_caption(self, model_name="llava"):
        mm_model = self.image_caption_models.get(model_name, None)
        if mm_model is None:
            raise ValueError(f"Model {model_name} is not registered.")
        
        dense_captions = self.load_result()
        for activity_name in tqdm(os.listdir(self.image_dir)[:2], desc='[Iter activities]'):
            activity_dir = os.path.join(self.image_dir, activity_name)
            for image_name in tqdm(os.listdir(activity_dir), desc='[Iter images]'):
                image_path = os.path.join(activity_dir, image_name)
                dense_caption = mm_model.get_caption(image_path, activity_name)
                if image_name in dense_captions:
                    continue
                dense_captions[image_name] = dense_caption
                print(f"Generate dense caption for {image_name}: {dense_caption}")
        
                # 及时保存
                if len(dense_captions) % self.save_interval == 0:
                    self.save_result(dense_captions)

        self.save_result(dense_captions)  # save the last batch
        print(f"Generate dense captions for {len(dense_captions)} images.")
        return dense_captions
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_dir", type=str, default="/training/wphu/Dataset/lavis/from_cap/")
    args = parser.parse_args()

    llava_model = LlavaModel()

    engine = GenDenseCaptionForClips(
        args.cap_dir
    )
    engine.register_model("llava", llava_model)
    engine.gen_dense_caption()
    print('Done')

