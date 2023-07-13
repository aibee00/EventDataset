#!/usr/bin/env bash
set -eo pipefail
set -x

USER=${1}
BRAND=${2}
CITY=${3}
BRANCH=${4}
DATE=${5}
work_path=${6}

# source ./config

hdfscli initkrb5 -k /home/${USER}/keytab/${USER}.keytab ${USER} ||
hdfscli initkrb5 -k /home/${USER}/${USER}.keytab ${USER} 

[[ -z $work_path ]] && work_path=`pwd`
cd $work_path
mkdir -p ./videos || echo "${work_path}/videos already exist!" 
cd ./videos
mkdir -p ${BRAND,,}_${CITY,,}_${BRANCH,,}_${DATE,,} || echo "${work_path}/videos/${BRAND,,}_${CITY,,}_${BRANCH,,}_${DATE,,} already exist!"
cd ${BRAND,,}_${CITY,,}_${BRANCH,,}_${DATE,,}

# Download both DIDs and FIDs videos to the same local path as following. The hdfs path of other benchmark videos are similar, please change the store site and date correspondingly.
echo "Downloading videos for ${BRAND,,}_${CITY,,}_${BRANCH,,}_${DATE,,}"
hdfscli download /bj_dev/prod/QA/qa_store/store_benchmarks/${BRAND}/${CITY}/${BRANCH}v7/${DATE}/processed/body/${DATE} ./ || echo "videos already exist!"
echo ln -s /ssd/wphu/videos/GACNE_guangzhou_xhthwk_20210717/ ${work_path}/videos/${BRAND,,}_${CITY,,}_${BRANCH,,}_${DATE,,}/${DATE} || echo "videos already exist!"


echo #################################################### \
# PLEASE DO NOT DOWNLOAD THE FOLLOWING VIDEOS SINCE  \
# THEY ARE NOT SPLITTED INTO 5 MIN SEGMENTS, AND NOT \
# USABLE IN THIS VERSION. PLAN TO FIX IT IN NEXT \
# VERSION. \
#################################################### 

echo hdfscli download /bj/archive/prod/customer/${BRAND}/${CITY}/${BRANCH}/videos/processed/face/${DATE} ./fids 
echo mv ./fids/* ./${DATE}
echo rm -rf ./fids

cd $cur_path
echo "Finish downloading videos, save path: ${work_path}/videos/"



