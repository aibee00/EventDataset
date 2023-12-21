#!/usr/bin/bash

: << 'Comment'
    Step1, You need to set EN_IOU_MATCH to False, then you can get all augmented images.
    Step2, You need to run detector with Aibee's detector to get all bboxes for all images. \
        This step need to run offline out of this script. Please refer to \
        http://wiki.aibee.cn/pages/viewpage.action?pageId=22731508 and https://aibee.feishu.cn/docs/doccnU3iGKbaQJFAK0cSKcciDNg .
        You can download the detector from `mms://singleview/bhrf_detection_store_retinanet_tf_gpu_general_20210419_v070000:v070000` .
    Step3, Run script `annotation_tools/parse_detections_from_weiding_result.py` to generate detections_result.json. 
        e.g. `python annotation_tools/parse_detections_from_weiding_result.py --input_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.txt --output_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json --use_provided_image_path`
    Step4, You need to run script `annotation_tools/gen_img_list.py` to generate img_list. 
    Step5, You need to set EN_IOU_MATCH to True and run this script, it will filter out matched bbox and generate corresponding labels.
    Step6, You need to run script `annotation_tools/convert_label_to_lavis_format.py` to generate final labels to train llava-style model in lavis.
Comment

# Step1
# 这一步是先生成所有扩增的样本，并拷贝所有样本的 images
python augment_imgs_for_label.py \
    --label_result_v1_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_en.json \
    --train_img_list_v1 /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json \
    --label_result_v2_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json \
    --img_dir /training/wphu/Dataset/lavis/eventgpt/images \
    --img_detections_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json \
    --save_dir /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand \
    --window 5 \
    --img_list_path /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/train_img_list_v2.json \
    --iou_threshold 0.5 \
    --Filter_NUM 413

# Step2 需要在 gpu=1080ti 的机器上运行
# sudo docker run --rm -it -v $PWD:/work registry.aibee.cn/mla/mmsctl:latest
# mmsctl download -u mms://singleview/bhrf_detection_store_retinanet_tf_gpu_general_20210419_v070000:v070000 -p download_model_dir  # download detector model
# sudo docker run -it --runtime=nvidia -v $PWD:/workspace --privileged registry.aibee.cn/mla/model-inference-x64-gpu:3.6.2  # cuda 10.0
# or sudo docker run -it --runtime=nvidia -v $PWD:/workspace --privileged registry.aibee.cn/mla/model-inference-x64-gpu-cuda114:3.6.2  # cuda 11.4
# eval_all  -m ./1080ti/ -i imglist.txt -r ./detection_result.txt


# 检查是否已经生成了detection_result.txt
if [ ! -f /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.txt ]; 
    echo "Please run detector first maually, refer to the following instructions:"
    echo "1. Find a IDC with GPU:1080ti, and mkdir a work dir for example named 'download_model_dir'"
    echo "2. Run docker: `sudo docker run --rm -it -v $PWD:/work registry.aibee.cn/mla/mmsctl:latest`"
    echo "3. Download detector model: `mmsctl download -u mms://singleview/bhrf_detection_store_retinanet_tf_gpu_general_20210419_v070000:v070000 -p download_model_dir`"
    echo "4. Run docker: "
    echo "  - If your cuda version is 10.0, run: `sudo docker run -it --runtime=nvidia -v $PWD:/workspace --privileged registry.aibee.cn/mla/model-inference-x64-gpu:3.6.2`"
    echo "  - If your cuda version is 11.4, run: `sudo docker run -it --runtime=nvidia -v $PWD:/workspace --privileged registry.aibee.cn/mla/model-inference-x64-gpu-cuda114:3.6`"
    echo "5. Cd your local path (download_model_dir) and run: `eval_all  -m ./1080ti/ -i imglist.txt -r ./detection_result.txt`"
    echo "detection_result.txt not found!"
    exit 1
fi

# Step3 
python annotation_tools/parse_detections_result.py \
    --input_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.txt \
    --output_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json \
    --use_provided_image_path

# Step4
python annotation_tools/gen_img_list.py \
    /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json  \
    /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/train_img_list_v2.json

# Step5
# 这一步主要是为了使用 iou 匹配过滤掉无效的样本，并且生成对应的 labels 文件(需要重新按照标注的img_list顺序排序)。
python augment_imgs_for_label.py \
    --label_result_v1_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_en.json \
    --train_img_list_v1 /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json \
    --label_result_v2_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json \
    --img_dir /training/wphu/Dataset/lavis/eventgpt/images \
    --img_detections_json /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json \
    --save_dir /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand \
    --window 5 \
    --img_list_path /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/train_img_list_v2.json \
    --iou_threshold 0.5 \
    --Filter_NUM 413 \
    --EN_IOU_MATCH \
    --RESORT_ACCORDING_TO_IMG_LIST

# Step6
python annotation_tools/convert_label_to_lavis_format.py \
    --use_augment


