""" Convert CH to EN caption and add dense caption to org label
"""
import os
import sys
import json
from tqdm import tqdm

# TASK 定义
TASK = {
    "STORE_INOUT": 1,
    "CAR_VISIT": 2,
    "CAR_INOUT": 3,
    "COMPANION": 4,
    "INDIVIDUAL_RECEPTION": 5,
    "UNKOWN": -1
}

input_file = sys.argv[1]
output_file = sys.argv[2]
dense_caption_file = sys.argv[3]


def convert_caption(caption):
    with open(input_file, 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        """ Convert caption to english
        """
        caption = item['caption']
        # caption = caption.replace('?', '? ')
        # caption = caption.replace('!', '! ')
        # caption = caption.replace('.', '. ')
        # caption = caption.replace(',', ', ')
        # caption = caption.replace('"', '" ')
        # caption = caption.replace("'", "' ")
        # caption = caption.replace('-', ' ')
        # caption = caption.replace('(', ' ')
        # caption = caption.replace(')', ' ')
        # caption = caption.replace('[', ' ')
        # caption = caption.replace(']', ' ')
        # caption = caption.replace(':', ' ')
        # caption = caption.replace(';', ' ')
        # caption = caption.replace('*', ' ')
        # caption = caption.replace('/', ' ')
        # caption = caption.replace('\\', ' ')
        # caption = caption.replace('`', ' ')
        # caption = caption.replace('~', ' ')
        # caption = caption.replace('<', ' ')
        # caption = caption.replace('>', ' ')
        # caption = caption.replace('%', ' ')
        # caption = caption.replace('#', ' ')
        # caption = caption.replace('&', ' ')
        # caption = caption.replace('_', ' ')
        # caption = caption.replace('+', ' ')
        # caption = caption.replace('=', ' ')
        # caption = caption.replace('{', ' ')
        # caption = caption.replace('}', ' ')
        # caption = caption.replace('|', ' ')
        # caption = caption.replace('^', ' ')
        # caption = caption.replace('°', ' ')
        # caption = caption.replace('§', ' ')
        # caption = caption.replace('¶', ' ')
        # caption = caption.replace('·', ' ')
        # caption = caption.replace('¿', ' ')
        # caption = caption.replace('¡', ' ')
        # caption = caption.replace('«', ' ')
        # caption = caption.replace('»', ' ')
        # caption = caption.replace('«', ' ')
        # caption = caption.replace('•', ' ')
        # caption = caption.replace('…', ' ')

        # convert main content
        caption = caption.replace('是同一批次', ' is the same group')
        caption = caption.replace('正进入该店大门', ' enters the door ')
        caption = caption.replace('正走出该店大门', ' exits the door ')
        caption = caption.replace('正在接待', ' is receiving ')
        caption = caption.replace('正在进入车', ' is entering the car ')
        caption = caption.replace('正在走出车', ' is exiting the car ')   
        caption = caption.replace('正在访问', ' is visiting ')

        item['caption'] = caption

    return data


def update_dense_caption(orignal_file, grit_cpation_file):
    """ Add grit caption to org data
    """
    # 判断orignal_file是否为路径
    if isinstance(orignal_file, str) and os.path.isfile(orignal_file):
        with open(orignal_file, 'r') as f:
            data = json.load(f)
    else:
        data = orignal_file
    
    with open(grit_cpation_file, 'r') as f:
        grit_data = json.load(f)
        grit_data = {item['image_id']: item['dense_caption'] for item in grit_data}


    for item in data:
        item['dense_caption'] = grit_data[item['image_id']]

    return data


def gen_prompt_by_task(data, max_len=50):
    """ Generate prompt by task
    """
    def padding_prompt(max_len, prompt):
        if len(prompt) < max_len:
            prompt = ' ' * (max_len - len(prompt)) + prompt  # pad left
        return prompt
    
    for item in tqdm(data):
        task = item['task']  # taks format: [TASK: n]
        
        if task == "[TASK: 1]":
            prompt = "STORE_INOUT events detecting: "
        elif task == "[TASK: 2]":
            prompt = "CAR_VISIT events detecting: "
        elif task == "[TASK: 3]":
            prompt = "CAR_INOUT events detecting: "
        elif task == "[TASK: 4]":
            prompt = "COMPANION events detecting: "
        elif task == "[TASK: 5]":
            prompt = "INDIVIDUAL_RECEPTION events detecting: "
        else:
            prompt = "UNKNOWN_TASK"
        
        max_len = len("INDIVIDUAL_RECEPTION events detecting: ")
        item['prompt'] = padding_prompt(max_len, prompt)
    return data

# write into json
data = convert_caption(input_file)
data = update_dense_caption(data, dense_caption_file)
data = gen_prompt_by_task(data)

if data:
    print(data[:10])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


