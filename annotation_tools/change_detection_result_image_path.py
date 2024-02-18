import json
import sys

input_file = sys.argv[1]  # /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json
output_file = sys.argv[2]  # /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand_person_index/detection_result_person_index.json

old_image_dir = sys.argv[3]  # /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images
new_image_dir = sys.argv[4]  # /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand_person_index/images

"""
"image": "/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images/hongqi_beijing_fkwd_20220109__ch01004_20220109204000__0516.jpg",
"bbox": "person:[0.023, 0.84, 0.132, 0.998];person:[0.034, 0.228, 0.109, 0.569];person:[0.081, 0.249, 0.151, 0.599];person:[0.084, 0.165, 0.148, 0.474];person:[0.149, 0.236, 0.181, 0.388];person:[0.171, 0.163, 0.213, 0.399];person:[0.27, 0.017, 0.303, 0.199];person:[0.295, 0.096, 0.343, 0.377];person:[0.308, 0.048, 0.342, 0.25];person:[0.361, 0.19, 0.431, 0.542];person:[0.381, 0.059, 0.421, 0.281];person:[0.423, 0.063, 0.465, 0.272];person:[0.467, 0.068, 0.505, 0.247];person:[0.481, 0.065, 0.527, 0.331];person:[0.515, 0.056, 0.561, 0.324];person:[0.552, 0.037, 0.593, 0.263];person:[0.713, 0.028, 0.756, 0.233];person:[0.794, 0.171, 0.855, 0.496]"
"""


def change_path_name(input_file, output_file, old_image_dir, new_image_dir):
    with open(input_file, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i]["image"] = data[i]["image"].replace(old_image_dir, new_image_dir)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    print(f"Input file: {input_file}")
    change_path_name(input_file, output_file, old_image_dir, new_image_dir)
    print("Done")
    print(f"Output file: {output_file}")

