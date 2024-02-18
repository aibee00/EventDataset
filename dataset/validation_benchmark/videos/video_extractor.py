""" Instruction
Auther: wphu
Date: 20240130
Description: Given events list, we need to extract video clips fro each events.
Note:
1. If we can not download videos successfully, we need to restore it first from OSS.
2. It must support new benchmark extensions just by providing site_id and date.
"""

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
sys.path.append(Path(__file__).parent.parent.parent.parent.as_posix())

import logging
import argparse
from typing import Optional

from dataset.validation_benchmark.events.build_event import DataArguments, EventBuilder
from dataset.validation_benchmark.videos.downloader import VideoDownloader
from video_config import VideoConfig
from cut_video import VideoCutter
import os
import pickle


def args_parser():
    parser = argparse.ArgumentParser(description="Download videos from eventgpt.")
    parser.add_argument("--bmk_path", type=str, default="/ssd/wphu/Benchmarks/", help="Path to benchmark.")
    parser.add_argument("--username", type=str, default="wphu", help="Username.")
    parser.add_argument("--keytab", type=str, default="/home/wphu/wphu.keytab", help="Path to keytab.")
    parser.add_argument("--hdfs_pid_output", type=str, 
                        default="/bj_dev/user/storesolution/LIXIANG/beijing/hjv7/car/images/20220715/pid_output_staff.tar", help="hdfs url to pid_output.")
    parser.add_argument("--camera_info_path", type=str, default="/ssd/wphu/CameraInfos/LIXIANG/beijing/hj/", help="Path to camera info.")
    parser.add_argument("--store_info_path", type=str, default="/ssd/wphu/StoreInfos/LIXIANG/beijing/hj/", help="Path to store info.")
    parser.add_argument("--site_id", type=str, default="LIXIANG_beijing_hj", help="site id")
    parser.add_argument("--date", type=str, default="20220715", help="date")
    parser.add_argument("--task_id", type=int, default=55, help="task id")
    parser.add_argument("--output_path", type=str, default="./tmp", help="Path to output.")
    return parser.parse_args()


class VideoProcessor(object):
    def __init__(self, 
            output_path: Optional[str], 
            eventbuilder: EventBuilder, 
            downloader: VideoDownloader,
            videocutter: VideoCutter,
        ):

        self.output_path = output_path
        self.eventbuilder = eventbuilder  # event loader
        self.downloader = downloader  # video downloader
        self.videocutter = videocutter  # video cutter
        
    def prepare_event_labels(self, site_id, date):
        labels = self.eventbuilder.load_events(site_id, date)
        return labels
    
    def get_first_event_label(self, site_id, date):
        labels = self.eventbuilder.labels
        if not labels:
            labels = self.eventbuilder.load_events(site_id, date)
            logging.warning(f"Please load event labels first! Current labels is None!")
            return None
        return labels[0]

    def process(self, site_id, date):
        # step1. prepare events
        labels_pkl = os.path.join(self.output_path, f"{site_id}_{date}_labels.pkl")
        if os.path.exists(labels_pkl):
            logging.info(f"Loading events from {labels_pkl}.")
            labels = pickle.load(open(labels_pkl, "rb"))
        else:
            labels = self.prepare_event_labels(site_id, date)
            pickle.dump(labels, open(labels_pkl, "wb"))

        events = self.eventbuilder.get_all_events(labels)

        # step2. download videos and cut video to clips according to events.
        for event in events:
            self.downloader.inject_config(VideoConfig.from_event(event, save_path=self.output_path))
            if not self.downloader.is_video_exist(site_id, date, event):
                logging.info(f"Video not exist, download it from {self.downloader.config.video_url}.")
                self.downloader.download_video(site_id, date, event)
            
            # 我们需要根据event对视频进行切片
            event_label = self.get_first_event_label(site_id, date)
            self.videocutter.cut_video(event_label, event, window_offset=3)


if __name__ == "__main__":
    args = args_parser()
    logging.basicConfig(level=logging.INFO)

    # step1. prepare eventbuilder and downloader
    eventbuilder = EventBuilder(
        bmk_path=args.bmk_path,  # default: /ssd/${USER}/Benchmarks
        output_path=args.output_path,
        data_args=DataArguments(
            username=args.username,
            keytab=args.keytab,  # default: /ssd/${USER}/${USER}.keytab
            hdfs_pid_output=args.hdfs_pid_output,
            camera_info_path=args.camera_info_path,
            store_info_path=args.store_info_path,
            task_id=args.task_id,
        )
    )

    downloader = VideoDownloader()

    # Init video cutter
    output_path = Path(args.output_path) / f"{args.site_id}_{args.date}" / "clips"
    videocutter = VideoCutter(
        video_meta_path=os.path.join(args.output_path, f"{args.site_id}_{args.date}", "meta"),
        output_path=output_path
    )

    # step2. process
    processor = VideoProcessor(args.output_path, eventbuilder, downloader, videocutter)
    processor.process(args.site_id, args.date)

    logging.info("Done.")
    sys.exit(0)

