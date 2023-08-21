import sys
import json

if __name__ == "__main__":
    # label json with bounding boxes from yolos
    input_file = sys.argv[1]  # e.g. label_train.json or label_test.json
    
    output_file = sys.argv[2]  # label json with only person/car/door bounding boxes from yolos

    key_words = ["person", "car", "door"]


    with open(input_file, "r") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        bbox = item["bbox"]

        bbox_f = ""
        objs = bbox.split('], ')

        objs = [o + ']' for o in objs if o and not o.endswith(']')]
        print(objs)

        for obj in objs:
            name, coord = obj.split(': ')

            for key_word in key_words:
                if key_word in name:
                    bbox_f += obj + '; '
            
        # update bbox
        item["bbox"] = bbox_f

    # save result to json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
