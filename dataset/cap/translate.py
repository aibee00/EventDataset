import os
import json
import csv


text_file = './dataset/cap/activities.txt'

json_file = './dataset/cap/activities.json'

output_file = './dataset/cap/activities.csv'


with open(text_file) as f:
    lines = f.readlines()

    # 去掉 index
    lines = [line.split('. ')[1] for line in lines]

    # to dict
    data = {}
    for line in lines:
        line = line.strip()
        en, cn = line.split(':')
        # 去掉en的双引号
        en = en.strip('"')
        data[en] = cn

with open(json_file, 'r') as f:
    data_json = json.load(f)

# save to csv
with open(output_file, 'w') as f:
    writer = csv.writer(f)
    # head
    writer.writerow(['en', 'cn', 'num'])
    for key, value in data_json.items():
        try:
            writer.writerow([key, data[key], value])
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()


