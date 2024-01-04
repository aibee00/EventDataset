import json
import os
import argparse
import shutil

from pathlib import Path
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image_dir", type=str, help="image path", \
                        default="/training/wphu/Dataset/lavis/eventgpt/images/")
    parser.add_argument("--label_dir", type=str, help="label result path", \
                        default="/training/wphu/Dataset/Tsinghua/eventgpt/BoxCaptionVQA/annotations/vqa_llava_style_box_caption_test.json")
    parser.add_argument("--output_dir", type=str, help="output", \
                        default="/training/wphu/Dataset/Tsinghua/eventgpt/BoxCaptionVQA/images/")
    parser.add_argument("--num_workers", type=int, help="num_workers", \
                        default=8)
    return parser.parse_args()


def main(args):
    # Parse label result and get all images
    labels = json.loads(open(args.label_dir, 'r').read())

    image_list = set([label['image'] for label in labels])

    save_dir = Path(args.output_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    params = []
    for image in tqdm(image_list):
        image_path = Path(args.source_image_dir) / image

        relative_path = Path(image).parent
        dest_path = save_dir / relative_path
        if not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / image
        if save_path.exists():
            # print(f"Skiping, Image {image} already exists.")
            continue

        params.append((image_path, save_path))

    if args.num_workers > 1:
        from multiprocessing import Pool
        with Pool(args.num_workers) as pool:
            pool.starmap(shutil.copy, params)

    else:
        for image_path, save_path in tqdm(params):
            shutil.copy(image_path, save_path)
            # print(f"Copying image {image} to {save_path}")

if __name__ == "__main__":
    args = get_args()
    main(args)
