#!/bin/bash
#!/usr/bin/env bash
set -eo pipefail
set -x

USER=${1}
BRAND=${2}
CITY=${3}
BRANCH=${4}
DATE=${5}
TAG=${6}
OUTPUT_PATH=${7}

benchmark_path="/ssd/${USER}/Benchmarks"

# gt path
gt_path1=$benchmark_path/${BRAND}/${CITY}/${BRANCH}/${DATE}/${BRAND}_${CITY}_${BRANCH}_${DATE}_converted_gt.json
gt_path2=$benchmark_path/${BRAND}/${CITY}/${BRANCH%v7*}/${DATE}/${BRAND}_${CITY}_${BRANCH%v7*}_${DATE}_converted_gt.json
gt_path3=$benchmark_path/${BRAND}/${CITY}/${BRANCH}v7/${DATE}/${BRAND}_${CITY}_${BRANCH}v7_${DATE}_converted_gt.json
if [[ -f $gt_path1 ]]; then
  gt_path=$gt_path1
elif [[ -f $gt_path2 ]]; then
  gt_path=$gt_path2
elif [[ -f $gt_path2 ]]; then
  gt_path=$gt_path3
fi
[ -f $gt_path ]


# event loader
python dataset/validation_benchmark/event_loader.py \
    --validation_label $gt_path \
    --output_path ${OUTPUT_PATH} 

