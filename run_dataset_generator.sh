#!/bin/bash

# USER=${1}
# BRAND=${2}
# CITY=${3}
# BRANCH=${4}
# DATE=${5}
# TAG=${6}

USER=wphu
BRAND=GACNE
CITY=guangzhou
BRANCH=xhthwk
DATE=20210717
TAG=v744.eventgpt

camera_info_path=/ssd/wphu/CameraInfos/${BRAND}/${CITY}/${BRANCH}/
store_infos_path=/ssd/wphu/StoreInfos/${BRAND}/${CITY}/${BRANCH}/
new_xy_path=/ssd/wphu/work_dir/${BRAND}-${CITY}-${BRANCH}-${DATE}-${TAG}/output_data/new_xy
car_pose=/ssd/wphu/work_dir/${BRAND}-${CITY}-${BRANCH}-${DATE}-${TAG}/inputs/pose.json
# events=/ssd/wphu/work_dir/${BRAND}-${CITY}-${BRANCH}-${DATE}-${TAG}/event_detection/events.pb
events=/ssd/wphu/Benchmark/${BRAND}/${CITY}/${BRANCH}/${DATE}/tpid_mappings.json
video_path=/ssd/wphu/videos/${BRAND}-${CITY}-${BRANCH}-${DATE}
dataset_path=/ssd/wphu/Dataset/${BRAND}-${CITY}-${BRANCH}-${DATE}

python dataset/dataset_generator.py \
    --camera_info_path $camera_info_path \
    --store_infos_path $store_infos_path \
    --car_pose $car_pose \
    --new_xy_path $new_xy_path \
    --events_file $events \
    --video_path $video_path \
    --dataset_path $dataset_path 




