import sys
import json
from pathlib import Path
import tqdm

input_file = sys.argv[1]  # vqa format json file
output_file = sys.argv[2]

IMG_SIZE = (1440, 2560)  # (H,W)

def denorm(box, img_size):
    H, W = img_size
    box[0] = int(box[0] * W)
    box[1] = int(box[1] * H)
    box[2] = int(box[2] * W)
    box[3] = int(box[3] * H)
    return box

def convert(input_file, output_file):
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    # new file
    with open(output_file, encoding='utf-8', mode='w') as f:
        # convert to dict {"content": "", "summary": ""}
        for data in raw_data:
            context = data.get("dense_caption")
            question = data.get("question")
            answer = data.get("answer")
            image = data.get("image")

            # denorm
            answer = eval(answer[0])
            answer = denorm(answer, IMG_SIZE)

            content = f"Context: {context}\nquestion: {question}\nanswer: "
        
            json.dump({"image": image, "content": content, "summary": answer}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    if not Path(output_file).parent.exists():
        Path(output_file).parent.mkdir(parents=True)
    
    convert(input_file, output_file)
