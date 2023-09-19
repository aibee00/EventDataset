import os
import os.path as osp
import argparse
from glob import glob
import random
from warnings import warn
import streamlit as st
import json


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path of raw images lib")
    parser.add_argument("--output_path", type=str, help="Dataset save path")
    parser.add_argument("--data_size", type=int, default=100, help="Number images of dataset")
    parser.add_argument("--ds_name", type=str, default="gacne_fowshot_data", 
                        help="Dataset name: which is also the name of the folder to save data")
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()
    return args


def tag_anno_to_img(labels: list, img: str, anno: str):
    label = {}
    label["img"] = img
    label["prompt"] = "这张图片的背景里有什么内容？"
    label["label"] = anno
    labels.append(label)


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


class DatasetGenerator(object):

    def __init__(self, root, output_path, data_size, ds_name="gacne_fowshot_data", share=False) -> None:
        """ 

        Parms:
            - root : root path of images
            - output : output path of dataset
        """
        self.root = root
        self.output_path = output_path
        self.data_size = data_size

        self.all_image_list = self._get_all_img_list()

        self.result_path = osp.join(output_path, 'dataset_img_list.json')
        self.result = self._get_indexs(data_size)

        self.dataset_name = ds_name
        self.share = share
        self.labels = []

    @property
    def max_capacity(self, ):
        return len(self.all_image_list)
    
    @property
    def len(self,):
        return self.__len__()

    def __len__(self,):
        return len(self.result)
    
    def __getitem__(self, item):
        img = self.result[item]
        
        # Copy img into dataset output path
        dataset_path = osp.join(self.output_path, self.dataset_name)
        if not osp.exists(dataset_path):
            os.makedirs(dataset_path)
        new_img = osp.join(dataset_path, osp.basename(img))
        
        print("Copying {} into {} ...".format(img, new_img))
        os.system("cp {} {}".format(img, new_img))

        return new_img
    
    def _get_all_img_list(self,):
        all_image_list = []

        # Get all image dir list
        data_folders = os.listdir(self.root)

        for folder in data_folders:
            cur_fold_imgs = glob(osp.join(self.root, folder, "*.jpg"))
            all_image_list.extend(cur_fold_imgs)
        
        return all_image_list
    
    def _get_indexs(self, data_size):
        """ Get all image index that will be used for generating dataset.

        Step:
            - Get all image dir list
            - Randomly choose data_size images from the image list
        Return : list(index)
        """
        # If result has existed, then load result
        # if osp.exists(self.result_path):
        #     self.result = json.loads(open(self.result_path, 'r').read())
        #     return self.result

        # Randomly choose data_size images from image_list
        result = []
        try:
            result = random.sample(self.all_image_list, data_size)
        except Exception as e:
            warn(f"The length of the data size {data_size} exceeds \
                          the maximum capacity of the img list {len(self.all_image_list)}, \
                          We will perform data sampling with replacement.")

            # Perform data sampling with replacement
            for i in range(data_size):
                result.append(random.choice(self.all_image_list))

        # save result
        with open(self.result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result
    
    def save_annotations(self, annotations, fname="dataset.json"):
        output_path = self.output_path

        if not osp.exists(output_path):
            os.makedirs(output_path)
        
        # Start annoations
        with open(osp.join(output_path, fname), 'w') as f:
            json.dump(annotations, f, indent=2)



def get_dataset_from_label(label_result, save_path):
    dataset = []

    results = json.loads(open(label_result, 'r').read())
    for img, labels in results.items():
        label = labels["label"]
        prompt = "这张图片的背景里有什么内容？"
        data = {
            "img" : img,
            "prompt" : prompt,
            "label" : label
        }
        dataset.append(data)
    
    # Save dataset
    if not osp.exists(save_path):
        os.makedirs(save_path)
    
    dataset_path = osp.join(save_path, "dataset.json")
    
    with open(dataset_path, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    

if __name__ == "__main__":
    args = argsparser()

    # Save dataset from label result
    label_result = './data/label_result.json'
    save_path = './data/'
    get_dataset_from_label(label_result=label_result, save_path=save_path)

    # ds_gen = DatasetGenerator(args.image_path, args.output_path, args.data_size, args.ds_name, args.share)
    
    # for i, img in enumerate(ds_gen):
    #     print(f"Finish generating {i}-th image {img} ...")
    
    # print(f"\nTotal {ds_gen.max_capacity} raw imgs in img_list, {ds_gen.len} be chosen to generate dataset.")




