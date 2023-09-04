import os
from multiprocessing import Pool
import cv2
from PIL import Image
import torch
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # Import ThreadPoolExecutor for thread pooling


from torchvision.transforms import functional as F
# from convert_to_coco_format_yolos_blip2 import YolosObjectDetection  # Assuming you have this class defined in a separate module
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class YolosObjectDetection():
    def __init__(self) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        if torch.cuda.is_available():
            self.model.cuda()
        
    def get_results(self, image):
        """
        Args:
            img_path: path of img in local
        Returns:
            dict: dict(tuple): tuple(score, bbox)
        """
        detections = []

        # image = Image.open(img_path)

        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs.to(self.model.device)
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label = self.model.config.id2label[label.item()]
            score = round(score.item(), 3)

            detections.append((label, score, box))

        # print detections
        # print(f"img: {img_path}, detections: {detections}")

        return detections
    
    @staticmethod
    def convert_context_description(detections):
        """ Convert detections into context description
        Args:
            detections: list(tuple): list(label, score, bbox)
        Returns:
            dict: context description
        """
        if not detections:
            return ""
        
        # convert to string format: "person: box, car: box, ..."
        context_str = ""
        for label, score, box in detections:
            context_str += f"{label}: {box}; "

        # 去掉最后的两个字符: ", "
        context_str = context_str[:-2]

        return context_str
    
    def detect(self, img_path):
        """
        Args:
            img_path: path of img in local
        Returns:
            dict: dict(tuple): tuple(score, bbox)
        """
        results = self.get_results(img_path)
        context = self.convert_context_description(results)

        return context


def process_video(video_path):
        # Create a folder for saving results
        video_name = os.path.basename(video_path).split('.')[0]
        output_folder = os.path.join(output_path, video_name)
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = 12
        print(f"Processing video: {video_path}, fps: {fps}")
        
        frame_num = 0
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cur_result = {}
            if frame_num % fps == 0:  # Process one frame per second
                image_name = f"{frame_num:04d}.jpg"
                image_path = os.path.join(output_folder, image_name)

                # 把frame转为Image格式
                image = Image.fromarray(frame)
                
                detections = detector.get_results(image)
                
                if any(label == 'person' for label, _, _ in detections):
                    # print(f"Video: {video_name}, Frame: {frame_num}, detections: {detections}")
                    os.makedirs(output_folder, exist_ok=True)
                    cv2.imwrite(image_path, frame)

                    context = detector.convert_context_description(detections)

                    cur_result.update({
                        "image": image_name,
                        "bbox": context
                    })
                    results.append(cur_result)

            frame_num += 1
        
        if os.path.exists(output_folder):
            print(f"video: {video_name}, frame: {frame_num}, saving results ...")
            output_folder_label = output_folder.replace("images", "labels")
            os.makedirs(output_folder_label, exist_ok=True)
            with open(os.path.join(output_folder_label, f"label_result_{frame_num}.json"), "w") as json_file:
                json.dump(results, json_file, indent=4)
        
        cap.release()


def collect_and_combine_labels(root_dir, label_path_name="labels"):
    combined_data = []

    def process_json(json_path, sub_dir):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        for item in data:
            relative_dir = sub_dir.split(label_path_name)[-1][1:]  # offset first char: '/'
            item['image'] = os.path.join(relative_dir, item['image'])
            combined_data.append(item)

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("label_result_") and file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                process_json(json_path, subdir)

    # Save the combined data to a new file
    combined_file_path = os.path.join(root_dir, "label_results.json")
    with open(combined_file_path, 'w') as combined_file:
        json.dump(combined_data, combined_file, indent=4)

    print("Processing and combining labels completed.")


def process_video_with_progress(parameters):
    video_path, overall_progress = parameters
    process_video(video_path)
    overall_progress.update()  # Update overall progress bar


if __name__ == "__main__":
    # Initialize the YolosObjectDetection instance
    detector = YolosObjectDetection()

    # Define the video paths
    video_paths = [
        "/ssd/wphu/Dataset/DatasetOnlyPerson/videos/vw_hefei_zl_20210727/20210727/ch01003_20210727214000.mp4.cut.mp4",
        "/ssd/wphu/Dataset/DatasetOnlyPerson/videos/vw_hefei_zl_20210727/20210727/ch01003_20210727162000.mp4.cut.mp4"
        # Add more video paths as needed
    ]

    output_path = "/ssd/wphu/Dataset/DatasetOnlyPerson/images/vw_hefei_zl_20210727/20210727"
    video_root = "/ssd/wphu/Dataset/DatasetOnlyPerson/videos/vw_hefei_zl_20210727/20210727"

    if ProcessAll:= True:
        video_paths = glob(os.path.join(video_root, "*.mp4.cut.mp4"))

    if UseProcess:= False:
        # ------------------------------ 进程实现 ------------------------------------
        # Determine the number of processes you want to use
        num_processes = 8  # Adjust as needed

        # Create a pool of processes
        with Pool(processes=num_processes) as pool:
            print(f"pool: {pool._pool}\n")
            # Create a tqdm instance to track overall progress
            overall_progress = tqdm(total=len(video_paths), desc="Videos processed", position=0, leave=True)
            
            parameters = [(path, overall_progress) for path in video_paths]
            # Start the processing using the pool of processes
            pool.map(process_video_with_progress, parameters)
            
            overall_progress.close()  # Close the overall progress bar
    
    elif UseThread:= False:
        # ------------------------------- 线程实现 -----------------------------------
        # Create a tqdm instance to track overall progress
        overall_progress = tqdm(total=len(video_paths), desc="Videos processed", position=0, leave=True)

        # Create a ThreadPoolExecutor with a limited number of threads
        max_threads = 2  # Set the maximum number of threads
        executor = ThreadPoolExecutor(max_workers=max_threads)

        # Use ThreadPoolExecutor to submit tasks
        future_to_path = {executor.submit(process_video_with_progress, (video_path, overall_progress)): video_path for video_path in video_paths}

        # Wait for all tasks to finish
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Threads completed", position=1, leave=True):
            pass

        overall_progress.close()  # Close the overall progress bar

    else:
        # ------------------------------- 单独进程完戽数 -----------------------------------
        for video_path in tqdm(video_paths, desc="Videos processed", position=0, leave=True):
            image_path = video_path.replace('videos', 'images').replace('.mp4.cut.mp4', '')
            print(f"image_path: {image_path}")
            if os.path.exists(image_path):
                print(f"The video {video_path} has been processed.")
                continue

            try:
                print(f"==Processing video in: {video_path}")
                process_video(video_path)
            except Exception as e:
                print(e)
                continue
        
    # 汇总所有的labels
    dataset_root = output_path.split('images')[0]
    labels_path = os.path.join(dataset_root, "labels")
    collect_and_combine_labels(labels_path)

    print("All videos processed.")
