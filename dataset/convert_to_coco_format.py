import os
import os.path as osp
import json
import sys
from tqdm import tqdm


input_file = sys.argv[1]
output_file = sys.argv[2]
is_val=sys.argv[3]

if is_val == 1:
    is_val = True
else:
    is_val = False

with open(input_file, 'r') as f:
    labels = json.load(f)

# convert
new_labels = []
for i, fname in enumerate(tqdm(labels)):
    label = labels[fname]

    if i == 0:
        print(label.keys())

    abs_path = label["img"]

    site_id = abs_path.split('/')[-3]
    relative_path = osp.join(site_id, fname)

    annotations = []
    annotations.append("描述:" + label["annotation"])
    annotations.append("背景信息:" + label["context"])
    if label.get("Car_context"):
        annotations[-1] += ";" + label["Car_context"]
    if label.get("Door_context"):
        annotations[-1] += ";" + label["Door_context"]

    if is_val:
        new_labels.append({
            "image": relative_path,
            "caption": annotations,  # list
            "image_id": fname.split('.')[0] 
        })
    else:
        new_labels.append({
            "image": relative_path,
            "caption": "##".join(annotations), # str
            "image_id": fname.split('.')[0]
        })


# save
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_labels, f, ensure_ascii=False, indent=2)


