""" Instruction
Auther: wphu
Date: 20240130
Description: Process of event come from validation benchmark.
    - Get all events list from the public platform.
    - Parse all GT events.

Stage1:
    - Get all events list from benchmark: https://code.aibee.cn/common/Benchmarks/-/tree/validation_v2
"""
import json
import logging
import argparse
from dataset.common import set_properties
from dataclasses import dataclass

from dataset.validation_benchmark.events.events import EventLabel
import os


def args_parser():
    parser = argparse.ArgumentParser(description="Process of event come from validation benchmark.")
    parser.add_argument("--bmk_path", type=str, default="/ssd/wphu/Benchmarks/", help="Path to benchmark.")
    parser.add_argument("--validation_label", type=str, default=None, help="Path to events validation label.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output.")
    return parser.parse_args()


@dataclass
class DataArguments:
    """ 基础信息的数据路径，如轨迹文件, camera_info_path, store_info_path等
    """
    username: str = None
    keytab: str = None
    hdfs_pid_output: str = None
    camera_info_path: str = None
    store_info_path: str = None
    task_id: int = None


class EventBuilder(object):
    """ Usage
    eventloader = EventBuilder(bmk_path, output_path, validation_label)
    eventloader.load_events(site, date)
    eventloader.get_events_by_type(event_type)
    eventloader.get_all_events()
    """
    def __init__(self, bmk_path, output_path, data_args: DataArguments, validation_label=None):
        self.bmk_path = bmk_path  # default: /ssd/${USER}/Benchmarks
        self.validation_label = validation_label
        self.output_path = output_path

        # data path arguments used for getting base info to init video config class
        self.data_args = data_args

        self.labels = None

    def _get_validation_label_path(self, site_id, date):
        """ Get validation label path by site_id and date.
        Args:
            site_id: str, site id. It must be format like this: "GANCE_guangzhou_xhthwk"
            date: str, date.
        Returns:
            path: str, path to validation label.
        """
        # site_id格式检查, 必须是两个下划线连接的三部分，如果是'-'连接则转成'_'
        if '-' in site_id:
            site_id = site_id.replace('-', '_')

        assert '_' in site_id, "site_id must be format like this: GANCE_guangzhou_xhthwk"

        brand, city, branch = site_id.split('_')

        # 这里要处理一下branch，因为它可能存在两种 {branch} 和 f'{branch}v7'
        if branch.endswith('v7'):
            branch = branch[:-2]
        
        path = os.path.join(self.bmk_path, brand, city, branch, date, 
                            f"{brand}_{city}_{branch}_{date}_converted_gt.json")
        
        if not os.path.exists(path):
            path = os.path.join(self.bmk_path, brand, city, f"{branch}v7", date,
                               f"{brand}_{city}_{branch}v7_{date}_converted_gt.json")
        return path

    def _load_label(self, site_id, date):
        # Get validation label path by site_id and date.
        if self.validation_label is None:
            self.validation_label = self._get_validation_label_path(site_id, date)

        with open(self.validation_label, "r") as f:
            label_list = json.load(f)
        return label_list

    def load_events(self, site_id, date):
        if self.labels is None:
            label_list = self._load_label(site_id, date)  # 这里之所以是个list是验证时间段会有多段
            logging.info(f"Loading events from {self.validation_label}")
            # update site_id into label
            label_list = [
                set_properties(
                    obj=label,
                    site_id=site_id,
                    username=self.data_args.username,
                    keytab=self.data_args.keytab,
                    hdfs_pid_output=self.data_args.hdfs_pid_output,
                    pid_output_save_path=os.path.join(self.output_path, f"{site_id}_{date}"),
                    camera_info_path=self.data_args.camera_info_path,
                    store_info_path=self.data_args.store_info_path,
                    task_id=self.data_args.task_id,
                )
                for label in label_list
            ]
            # Construct Event Label Structure
            self.labels = [
                EventLabel.from_dict(label) for label in label_list
            ]
            logging.info(f"Events loaded from {self.validation_label}")
        return self.labels
    
    def get_events_by_type(self, event_type="STORE_INOUT"):
        """ Get events by event type.
        Args:
            event_type: str, event type.
        Returns:
            events: List[BaseEvent], list of events.
        """
        assert self.labels is not None, "Please load events first by method self.load_events."

        label_list = self.labels
        events = []  # 这里之所以是个列表是因为验证会出现多个时间段如14:00-15:00, 15:00-16:00, 16:00-17:00等等
        for event_info in label_list:
            for etype, event in event_info.labels.items():
                if etype == event_type:
                    events.extend(event)
        return events
    
    def get_all_events(self, labels=None):
        """ Get all events.
        Return all events. Dict[event_type, List[Events]].
        """
        if labels is not None:
            self.labels = labels
        
        assert self.labels is not None, "Please load events first by method self.load_events."

        label_list = self.labels
        events = []
        for event_info in label_list:
            # event_info 是一个 EventLabel 对象
            for etype, event_list in event_info.labels.items():
                events.extend(event_list)
        return events


if __name__ == "__main__":
    args = args_parser()

    event_builder = EventBuilder(
        bmk_path=args.bmk_path,  # default: /ssd/${USER}/Benchmarks
        output_path=args.output_path,
        data_args=DataArguments(
            username="wphu",
            keytab="/ssd/wphu/keytab/wphu.keytab",
            hdfs_pid_output="/bj_dev/user/storesolution/LIXIANG/beijing/hjv7/car/images/20220715/pid_output_staff.tar",
            camera_info_path="/ssd/wphu/CameraInfos/LIXIANG/beijing/hj/20220715/",
            store_info_path="/ssd/wphu/StoreInfos/LIXIANG/beijing/hj/20220715/",
            task_id=55
        )
    )
    event_builder.load_events(site_id="LIXIANG_beijing_hj", date="20220715")
    store_inout_events = event_builder.get_events_by_type(event_type="STORE_INOUT")
    individual_reception_events = event_builder.get_events_by_type(event_type="INDIVIDUAL_RECEPTION")
    room_inout_events = event_builder.get_events_by_type(event_type="ROOM_INOUT")

    print("Store Inout Events:\n")
    for event in store_inout_events:
        print(event)
    
    print("Individual Reception Events:\n")
    for event in individual_reception_events:
        print(event)

    print("Room Inout Events:\n")
    for event in room_inout_events:
        print(event)

