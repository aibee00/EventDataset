#!/bin/bash
#!/usr/bin/env bash

######################################
# 准备数据
# 1. GT的new_xy.tar.gz
# 2. GT的pid_output.tar
# 3. car_pose文件: pose.json
# 4. videos
######################################

set -eo pipefail
set -x

USER=${1}
BRAND=${2}
CITY=${3}
STORE=${4}
DATE=${5}
LOCAL_BASE=${6}  # 保存这些数据的root路径

[[ -z ${LOCAL_BASE} ]] && echo "need LOCAL_BASE" && exit 1 

[[ ! -d ${LOCAL_BASE} ]] && mkdir -p ${LOCAL_BASE}

KEYTAB=/home/${USER}/${USER}.keytab
hdfscli initkrb5 -k ${KEYTAB} ${USER};

# 1. GT的new_xy.tar.gz
NEW_XY_HDFS_PATH=/bj_dev/user/wphu/gt_new_xy.final.tar
hdfscli download ${NEW_XY_HDFS_PATH} ${LOCAL_BASE}/gt_new_xy.tar
tar xvf ${LOCAL_BASE}/gt_new_xy.tar -C ${LOCAL_BASE}

# 2. GT的pid_output.tar
DATA_PATH=/bj_dev/user/store_solutions/store_benchmarks/${BRAND}/${CITY}/${STORE}v7/${DATE}/pid_output.tar
DATA_PATH=/bj_dev/user/store_solutions/store_benchmarks/${BRAND}/${CITY}/${STORE}v7/${DATE}/gtevents/pid_output.tar
hdfscli download ${DATA_PATH} ${LOCAL_BASE}/pid_output.tar
tar -xvf ${LOCAL_BASE}/pid_output.tar -C ${LOCAL_BASE}


# 3. car_pose文件: pose.json
DATA_PATH=/bj_dev/user/store_solutions/store_benchmarks/${BRAND}/${CITY}/${STORE}v7/${DATE}/pose.json
hdfscli download ${DATA_PATH} ${LOCAL_BASE}/pose.json

# 4. videos
# bash download_videos.sh ${USER} ${BRAND} ${CITY} ${STORE} ${DATE} ${LOCAL_BASE}





