import sys
import json


if __name__ == '__main__':
    input_file_yolos = sys.argv[1]
    input_file_label = sys.argv[2]

    with open(input_file_label, 'r') as f:
        data_label = json.load(f)

    with open(input_file_yolos, 'r') as f:
        data_yolos = json.load(f)

    for i, (label, yolos) in enumerate(zip(data_label, data_yolos)):
        assert label['image'] == yolos['image']
        assert label['task'] == yolos['task']
        # assert label['caption'] == yolos['caption'], f'caption not match {label["caption"]} vs {yolos["caption"]}'

        label['bbox'] = yolos['bbox']

    # overwite input_file_label
    with open(input_file_label, encoding='utf-8', mode='w') as f:
        json.dump(data_label, f, ensure_ascii=False, indent=4)

    print(f"Done! {input_file_label} updated.")
        


