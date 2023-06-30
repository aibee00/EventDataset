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
VER=v3

##### 版本说明 #####
# - v0: 把Prompt描述更新到label
# - v1: 加入bbox的位置编码(参考SegmentAnthing)
# - v2: 加入bbox的归一化(参考LLaVA)
# - v3: 综合V0,V1,V2

camera_info_path=/ssd/${USER}/CameraInfos/${BRAND}/${CITY}/${BRANCH}/
store_infos_path=/ssd/${USER}/StoreInfos/${BRAND}/${CITY}/${BRANCH}/

################ 获取 new_xy ###############
# 这里需要用GT的new_xy 
# new_xy_path=/ssd/${USER}/work_dir/${BRAND}-${CITY}-${BRANCH}-${DATE}-${TAG}/output_data/new_xy
# ROOT=/ssd/${USER}/work_dir
ROOT=`pwd`/data/${BRAND}-${CITY}-${BRANCH}-${DATE}
new_xy_path=$ROOT/gt_new_xy

# 判断是否存在，如果不存在则从HDFS下载
[[ -e $new_xy_path ]] || bash dataset/download_new_xy.sh $USER $BRAND $CITY $BRANCH $DATE $ROOT

new_xy_path_tar=$new_xy_path/${BRAND}-${CITY}-${BRANCH}-${DATE}/gt_new_xy.tar.gz
new_xy_path=$new_xy_path/${BRAND}-${CITY}-${BRANCH}-${DATE}/new_xy

# 判断new_xy_path是否存在，如果$new_xy_path不存在且$new_xy_path_tar存在则创建$new_xy_path，并解压到$new_xy_path
[[ -e $new_xy_path ]] || (mkdir -p $new_xy_path && tar xvf $new_xy_path_tar -C $new_xy_path)
#################################################

################ 获取 pid_output ###############
pid_output_path=`pwd`/data/${BRAND}-${CITY}-${BRANCH}-${DATE}/pid_output
[[ -e $pid_output_path ]] || bash dataset/prepare_data.sh $USER $BRAND $CITY $BRANCH $DATE `dirname $pid_output_path`
#################################################

car_pose=${ROOT}/pose.json
# events=/ssd/${USER}/work_dir/${BRAND}-${CITY}-${BRANCH}-${DATE}-${TAG}/event_detection/events.pb
events=/ssd/${USER}/Benchmarks/${BRAND}/${CITY}/${BRANCH}v7/${DATE}/tpid_mappings.json
video_path=/ssd/${USER}/videos/${BRAND}-${CITY}-${BRANCH}-${DATE}
dataset_path=/ssd/${USER}/Dataset/${BRAND}-${CITY}-${BRANCH}-${DATE}

python dataset/dataset_generator.py \
    --camera_info_path $camera_info_path \
    --store_infos_path $store_infos_path \
    --car_pose $car_pose \
    --new_xy_path $new_xy_path \
    --pid_output_path $pid_output_path \
    --events_file $events \
    --video_path $video_path \
    --dataset_path $dataset_path \
    --version $VER




