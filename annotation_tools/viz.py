import argparse
import glob
import logging
import os
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
mlog = logging.getLogger("myLogger")
level = logging.getLevelName("INFO")
mlog.setLevel(level)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file", required=True)
    parser.add_argument("--img_path", type=str, help="img path", required=True)
    parser.add_argument("--output_path", type=str, help="output path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(line.strip())
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for line in data:
        line = line.split(" ")
        img_name = os.path.basename(line[0])
        line = line[1:]
        assert len(line) % 8 == 0
        img_file = os.path.join(args.img_path, img_name)
        assert os.path.exists(img_file)
        img = cv2.imread(img_file)
        num = int(len(line) / 8)
        for i in range(num):
            cls = line[8 * i + 1]
            if cls != "1":
                continue
            xywh = line[8 * i + 2: 8 * i + 6]
            x1, y1, w, h = map(int, xywh)
            x2 = x1 + w
            y2 = y1 + h
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.putText(img, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        out_file = os.path.join(args.output_path, img_name)
        cv2.imwrite(out_file, img)

        