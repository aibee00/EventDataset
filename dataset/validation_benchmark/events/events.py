""" Instruction
Define data class for each events.
"""
import logging
import time

from tqdm import tqdm
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Any
from aibee_hdfs import hdfscli

from dataset.validation_benchmark.events.base_event import BaseEvent
from dataset.validation_benchmark.trajectory.track import TrackLoader
from dataset.validation_benchmark.registry import registry
from dataset.common import get_cover_channels, get_nearest_slice_start_time, string_to_ts
from dataset.gen_grid_cameras_map import gen_grid_cameras_map

# ------------------------- Event -----------------------
""" 这里可能label里名字只有后面这5种
EVENT_NAME_CONVERSION = {
        'REGION_INOUT': 'ROOM_INOUT',
        'CAR_INOUT': 'ROOM_INOUT',
        'CAR_VISIT': 'REGION_VISIT',
        'INDIVIDUAL_RECEPTION': 'INDIVIDUAL_RECEPTION',
        'STORE_INOUT': 'STORE_INOUT',
        'COUNTER_VISIT':  'COUNTER_VISIT'
        }
"""
@dataclass
@registry.register_event("STORE_INOUT")
class StoreInoutEvent(BaseEvent):
    GroupID: Optional[str] = field(default="")
    InDoorID: Optional[str] = field(default="")
    OutDoorID: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "STORE_INOUT"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__
        

@dataclass
@registry.register_event("ROOM_INOUT")
class RoomInout(BaseEvent):
    RegionID: Optional[int] = field(default=1)
    RegionType: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "ROOM_INOUT"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__


@dataclass
@registry.register_event("CAR_INOUT")
class CarInout(BaseEvent):
    RegionID: Optional[int] = field(default=1)
    RegionType: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "CAR_INOUT"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__


@dataclass
@registry.register_event("REGION_VISIT")
class RegionVisitEvent(BaseEvent):
    RegionID: Optional[int] = field(default=1)
    RegionType: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "REGION_VISIT"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__


@dataclass
@registry.register_event("COUNTER_VISIT")
class CounterVisitEvent(BaseEvent):
    RegionID: Optional[int] = field(default=1)
    RegionType: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "COUNTER_VISIT"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__


@dataclass
@registry.register_event("INDIVIDUAL_RECEPTION")
class IndividualReceptionEvent(BaseEvent):
    StaffID: Optional[str] = field(default="")
    Type: str = "INDIVIDUAL_RECEPTION"
    Comment: Optional[str] = field(default="")

    def __post_init__(self):
        self.Type = "INDIVIDUAL_RECEPTION"  # 设置默认的 Type
        super().__post_init__()  # 调用基类的 __post_init__


# ------------------------- Sampling -----------------------
@dataclass
class SamplingStruct:
    sampling: Optional[str] = field(default="select_time")
    sampling_range: Union[List[str], List[List[str]]] = field(default_factory=list)

    def __post_init__(self):
        for time_str in self.sampling_range:
            try:
                # 尝试将字符串解析为时间
                datetime.strptime(time_str, '%H:%M:%S')
            except ValueError:
                # 如果字符串不符合时间格式，抛出异常
                raise ValueError(f"Invalid time format: {time_str}")


# ---------------------- Event Label ------------------------
@dataclass
class HdfsClient:
    username: str = "wphu"
    keytab: str = "/home/wphu/wphu.keytab"
    _client: hdfscli.HdfsClient = field(default=None, init=False, repr=False, compare=False, hash=False)

    @property
    def client(self):
        if self._client is None:
            hdfscli.initKerberos(self.keytab, self.username)
            self._client = hdfscli.HdfsClient(self.username)
        return self._client


@dataclass
class MixinData(HdfsClient):
    hdfs_pid_output: str = ""
    pid_output_save_path: str = ""
    camera_info_path: Optional[str] = field(
        default="", 
        metadata={"help": "需要包含site_id和date, e.g. /ssd/wphu/CameraInfos/LIXIANG/beijing/hj/20220715"}
    )
    store_info_path: Optional[str] = field(
        default="", 
        metadata={"help": "需要包含site_id和date, e.g. /ssd/wphu/StoreInfos/LIXIANG/beijing/hj/20220715"}
    )

    track_loader: Optional[TrackLoader] = field(default_factory=TrackLoader)
    grid_cameras_map: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        # 假设 TrackLoader 和 gen_grid_cameras_map 是已经定义好的函数或类
        # 这里，我们在 __post_init__ 中使用 camera_info_path 和 store_info_path 的值来初始化其他属性
        self.track_loader = TrackLoader(self.client, self.hdfs_pid_output, self.pid_output_save_path)
        logging.info(f"Generating grid cameras map ...")
        tic = time.time()
        self.grid_cameras_map = gen_grid_cameras_map(self.camera_info_path, self.store_info_path)
        toc = time.time()
        logging.info(f"Grid cameras map generated. Total time: {toc - tic:.2f} seconds.")

    def _get_time_slices(self, event, interval=15):
        """
        Get time slice start time.
        Args:
            event: BaseEvent.
            interval: int. e.g. 15 minutes
        Returns:
            time_slices: List[str]. e.g. ["19:00:00", "20:00:00"]
        """
        date = datetime.strptime(self.date, "%Y%m%d").date()
        start_time = datetime.combine(date, event.StartTime)
        end_time = datetime.combine(date, event.EndTime)

        # 计算距离给定时间最近的整15分钟点
        nearst_interval_start_time = get_nearest_slice_start_time(start_time, interval)
        nearst_interval_start_time = datetime.combine(date, nearst_interval_start_time)

        time_slices = []
        start_ptr = nearst_interval_start_time
        while start_ptr <= end_time:
            time_slices.append(start_ptr.time().strftime('%H:%M:%S'))
            start_ptr += timedelta(minutes=interval)
        return time_slices
    
    @staticmethod
    def get_oss_video_name(date, best_channel, time_slice_start):
        """ 获取保存在 oss 上的视频的名字
        exp: ch01001_20230906190000.mp4.cut.mp4.down.mp4
        """
        return f"{best_channel}_{date}{time_slice_start.replace(':', '')}.mp4.cut.mp4.down.mp4"
    
    def get_best_channel(self, event):
        """ Get best coverage camera channel.
        Args:
            track_loader: instance of TrackLoader
        Returns:
            best_channel: str. e.g. "ch01002"
        """
        locs = self.track_loader.get_locs(event.Pid)

        assert locs, f"Error! Can not get valid locs of {event.Pid}."

        def to_ts_time_slice(event):
            start_ts = string_to_ts(event.StartTime)
            end_ts = string_to_ts(event.EndTime)
            return [start_ts, end_ts]
        
        time_slice = to_ts_time_slice(event)

        channels = get_cover_channels(
            grid_cameras_map=self.grid_cameras_map,
            pid=event.Pid,
            loc=locs,
            time_slice=time_slice,
        )
        
        if channels:
            return channels[0]
        return None  # 返回 None，表示没有找到最佳覆盖通道
    
    def get_best_channel_by_ts(self, pid, ts):
        """ Get best coverage camera channel.
        Args:
            track_loader: instance of TrackLoader
        Returns:
            best_channel: str. e.g. "ch01002"
        """
        locs = self.track_loader.get_locs(pid)

        assert locs, f"Error! Can not get valid locs of {pid}."

        channels = get_cover_channels(
            grid_cameras_map=self.grid_cameras_map,
            pid=pid,
            loc=locs,
            time_slice=ts,
        )

        if channels:
            return channels[0]
        return None  # 返回 None，表示没有找到最佳覆盖通道


@dataclass
class EventLabel(MixinData):
    site_id: Optional[str] = field(default="")
    date: Optional[str] = field(default="")
    task_id: Optional[int] = field(default=0) 
    start_time: Optional[str] = field(default="")
    end_time: Optional[str] = field(default="")
    ignore_events: Dict[str, List[BaseEvent]] = field(default_factory=dict)
    labels: Dict[str, List[BaseEvent]] = field(default_factory=dict)
    verification_sampling_method: Dict[str, SamplingStruct] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for key, value in self.ignore_events.items():
            for event in value:
                event.Type = key
        for key, value in self.labels.items():
            for event in value:
                event.Type = key
        for key, value in self.verification_sampling_method.items():
            value.sampling = key
        self.start_time = datetime.strptime(self.start_time, '%H:%M:%S').time()
        self.end_time = datetime.strptime(self.end_time, '%H:%M:%S').time()

        self.ignore_events = {key: [event for event in value if isinstance(event, BaseEvent)] for key, value in
                              self.ignore_events.items()}
        self.labels = {key: [event for event in value if isinstance(event, BaseEvent)] for key, value in
                       self.labels.items()}
        self.verification_sampling_method = {key: value for key, value in
                                              self.verification_sampling_method.items() if
                                              isinstance(value, SamplingStruct)}
    
    
    def construct_verification_sampling_method(self, verification_sampling_method):
        for event_name, item in verification_sampling_method.items():
            self.verification_sampling_method[event_name] = SamplingStruct(**item)

    def construct_ignore_events(self, ignore_events):
        for event_name, item in ignore_events.items():
            event_class = registry.get_event_class(event_name)
            if event_class is None:
                raise ValueError(f"Unknown event type: {event_name}")
            self.ignore_events[event_name] = [event_class(**event) for event in item]

            # update new extra properties
            logging.info(f"[construct_ignore_events] Updating new extra properties for {event_name} ...")
            tic = time.time()
            valid_events = []
            for event in self.ignore_events[event_name]:
                best_ch = self.get_best_channel(event)

                # 找不到相机覆盖，则丢掉该event
                if not best_ch:
                    logging.warning(f"Error! Can not find best coverage channel for {event.Pid}.")
                    continue  # 丢掉该event，继续下一个event

                time_slices = self._get_time_slices(event)  # oss上视频是以15分钟切片的，一个event包含多个视频片段
                for time_slice_start in tqdm(time_slices, desc='[construct_ignore_events] Iter time slices'):
                    event.set_properties(
                        site_id=self.site_id, 
                        date=self.date, 
                        video_name=self.get_oss_video_name(self.date, best_ch, time_slice_start),
                        channel_id=best_ch,
                        task_id=self.task_id
                    )
                valid_events.append(event)
            # update events
            self.ignore_events[event_name] = valid_events

            toc = time.time()
            logging.info(f"[construct_ignore_events] Updated new extra properties for {event_name}. Total time: {toc - tic:.2f} seconds.")

    def construct_labels(self, labels):
        """
        "labels": {
            "ROOM_INOUT": [
            {
                "EndTime": "14:17:20",
                "Pid": "N000110_140840",
                "RegionID": 1,
                "RegionType": "CAR",
                "StartTime": "14:09:07",
                "Type": "ROOM_INOUT",
                "issue_category": "",
                "issue_details": "",
                "record": "TP"
            },
            {
                "EndTime": "14:17:25",
                "Pid": "N000109_140840",
                "RegionID": 1,
                "RegionType": "CAR",
                "StartTime": "14:12:48",
                "Type": "ROOM_INOUT",
                "issue_category": "",
                "issue_details": "",
                "record": "TP"
            },...
        }
        """
        for event_type, item in labels.items():
            event_class = registry.get_event_class(event_type)
            if event_class is None:
                raise ValueError(f"Unknown event type: {event_type}")
            self.labels[event_type] = [event_class(**event) for event in item]

            # update new extra properties
            logging.info(f"[construct_labels] Updating new extra properties for {event_type} ...")
            tic = time.time()
            valid_events = []
            for event in self.labels[event_type]:
                best_ch = self.get_best_channel(event)

                # 找不到相机覆盖，则丢掉该event
                if not best_ch:
                    logging.warning(f"Error! Can not find best coverage channel for {event.Pid}.")
                    continue  # 丢掉该event，继续下一个event

                time_slices = self._get_time_slices(event)  # oss上视频是以15分钟切片的，一个event包含多个视频片段
                for time_slice_start in tqdm(time_slices, desc='[construct_labels] Iter time slices'):
                    event.set_properties(
                        site_id=self.site_id, 
                        date=self.date, 
                        video_name=self.get_oss_video_name(self.date, best_ch, time_slice_start),
                        channel_id=best_ch,
                        task_id=self.task_id
                    )
                valid_events.append(event)
            self.labels[event_type] = valid_events
            
            toc = time.time()
            logging.info(f"[construct_labels] Updated new extra properties for {event_type}. Total time: {toc - tic:.2f} seconds.")

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an EventLabel instance from a dictionary.

        Args:
            data: dict, 这里的data是gt_label中的信息和数据路径的merge.
            这里的主要数据路径参数需要有:
                - new_xy_path
                - camera_info_path
                - store_info_path
        
        Note:
            如果数据中没有嵌套结构，可以直接使用 cls(**data) 创建实例
            如果数据中有嵌套结构，需要先创建一个空实例，然后再递归地构建嵌套结构
            这里使用 cls(**data) 创建空实例，但是实际上是创建一个 EventLabel 实例，但暂时不包含复杂的嵌套结构
        """
        logging.info("Creating EventLabel instance from dictionary...")
        # 创建一个 EventLabel 实例，但暂时不包含复杂的嵌套结构
        instance = cls(
            # 添加hdfscli相关参数
            username=data.get("username", ""),
            keytab=data.get("keytab", ""),
            # 添加数据路径参数
            hdfs_pid_output=data.get("hdfs_pid_output", ""),
            pid_output_save_path=data.get("pid_output_save_path", ""),
            camera_info_path=data.get("camera_info_path", ""),
            store_info_path=data.get("store_info_path", ""),
            # 添加其他参数
            site_id=data.get("site_id", ""),
            date=str(data.get("date", "")),
            task_id=data.get("task_id", 0),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            # 默认值是空字典或列表
            ignore_events={},
            labels={},
            verification_sampling_method={}
        )

        # 分别构建复杂的嵌套结构
        if "verification_sampling_method" in data:
            instance.construct_verification_sampling_method(data["verification_sampling_method"])
        if "ignore_events" in data:
            instance.construct_ignore_events(data["ignore_events"])
        if "labels" in data:
            instance.construct_labels(data["labels"])

        # 返回完全构建的实例
        logging.info("EventLabel instance created successfully!")
        return instance
