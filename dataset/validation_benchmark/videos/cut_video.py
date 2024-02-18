import logging
import os
import json
import datetime
from datetime import datetime as dt
from datetime import timedelta as td

from pathlib import Path
import re
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from dataset.validation_benchmark.events.base_event import BaseEvent
from dataset.common import get_nearest_slice_start_time, string_to_ts, ts_to_string
from dataset.validation_benchmark.events.events import EventLabel


class VideoCutter(object):
    """ 将本地的15分钟一段的视频，按照给定的event时间进行切片
    """
    def __init__(self, video_meta_path, output_path):
        self.video_meta_path = video_meta_path
        self.output_path = output_path

        self.video_meta_infos = self._load_video_meta()

        self.events_just_focus_head_tail = ["STORE_INOUT", "CAR_INOUT", "ROOM_INOUT"]

    def _load_video_meta(self):
        """ Load video meta.
        Returns:
            video meta.
        """
        video_meta_files = Path(self.video_meta_path).glob("*.json")

        video_meta_infos = {}
        for video_meta_file in video_meta_files:
            with open(video_meta_file, "r") as f:
                video_meta = json.load(f)
            file_name = Path(video_meta_file).stem
            video_meta_infos[file_name] = video_meta
        return video_meta_infos
        
    def get_video_path(self, video_name):
        """ Get video path.
        Returns:
            video path.
        """
        return self.video_meta_infos[video_name]["video_path"]
    
    def get_video_name_by_ts(self, event, ts, best_channel=None):
        """ Get video name by ts.
        Args:
            event: event object.
            ts: timestamp.
            best_channel: best channel. 这里因为重新计算了start/end 时刻的best_ch，所以要update.
        Returns:
            video name.
        """
        video_start_time = get_nearest_slice_start_time(ts)  # datetime.time
        org_video_name = event.video_name
        video_start_time_index = org_video_name.find(event.date) + len(event.date)
        org_start_time = org_video_name[video_start_time_index: video_start_time_index + 6]
        video_name = org_video_name.replace(org_start_time, str(video_start_time.strftime("%H%M%S")))

        # update best channel
        if best_channel is not None:
            ch_idx = video_name.find("ch")
            if ch_idx == -1:
                raise ValueError(f"Invalid video name: {org_video_name}, it must contain a channel.")
            org_ch = video_name[ch_idx: ch_idx + len(best_channel)]
            video_name = video_name.replace(org_ch, best_channel)
        
        return video_name

    @staticmethod
    def extract_start_time_from_filename(filename):
        """
        从给定的视频文件名中提取开始时间。
        假设时间格式遵循 'YYYYMMDDHHMMSS' 的模式，位于文件名的末尾部分。
        
        Args:
        - filename: 视频文件的路径名
        
        Returns:
        - 开始时间的字符串表示，格式为 'HHMMSS'，如果未找到合适的时间则返回 None
        """
        # 使用正则表达式匹配路径中的时间部分
        try:
            match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{6})', filename)
            # 返回匹配到的时间部分，这里假设是 'HHMMSS' 格式
            return match.group(4)
        except:
            raise ValueError(f"Invalid filename: {filename}, it must contain a time in the format YYYYMMDDHHMMSS.")
        
    def _get_actual_start_end_time(self, event):
        """ 计算实际的切片时间，每个15视频的开始时间相对秒数
            根据event的start/end时间段进行切片.
        Args:
            event: event object.
        Returns:
            start_time, end_time.
        """
        # 从文件名提取视频开始时间
        extracted_time_str = self.extract_start_time_from_filename(event.video_name)
        
        # 将提取的时间转换为秒
        extracted_time = datetime.datetime.strptime(extracted_time_str, "%H%M%S").time()
        video_start_seconds = extracted_time.hour * 3600 + extracted_time.minute * 60 + extracted_time.second
        
        # 将事件开始和结束时间转换为秒
        event_start_time = event.StartTime.hour * 3600 + event.StartTime.minute * 60 + event.StartTime.second
        event_end_time = event.EndTime.hour * 3600 + event.EndTime.minute * 60 + event.EndTime.second
        
        # 计算实际剪辑的开始和结束时间（秒）
        actual_start_time = max(0, event_start_time - video_start_seconds)
        actual_end_time = event_end_time - video_start_seconds

        return actual_start_time, actual_end_time
    
    def _get_video_start_end_seconds(self, date, video_name):
        """ 根据video的名字获取，video开始时间，进而得到start/end time seconds
        Args:
            event: event object.
        Returns:
            start_time, end_time.
        """
        # 从文件名提取视频开始时间
        video_start_time_str = self.extract_start_time_from_filename(video_name)  # e.g.185959
        
        # 将视频开始时间转换为秒
        video_start_time = dt.strptime(video_start_time_str, "%H%M%S").time()
        date = dt.strptime(date, "%Y%m%d").date()
        video_start_time = dt.combine(date, video_start_time)
        video_end_time = video_start_time + td(seconds=15 * 60)
        video_start_seconds = video_start_time.hour * 3600 + video_start_time.minute * 60 + video_start_time.second
        video_end_seconds = video_end_time.hour * 3600 + video_end_time.minute * 60 + video_end_time.second
        return video_start_seconds, video_end_seconds
    
    def get_actual_start_end_window_time(self, event_label, event, offset=5):
        """ 计算实际的切片时间，每个15视频的开始时间相对秒数
            根据event的start/end时刻的一个window内进行切片
            返回actual windows time列表
            window: [event_start - offset, event_start + offset] or [event_end - offset, event_end + offset]
        
        应用于事件: 
            - STORE_INOUT
            - CAR_INOUT
            - ROOM_INOUT
        
        Args:
            event_label: event_label object. 第一个验证的时间段的 EventLabel 实例
            event: event object.
        
        Returns:
            windows: List[tuple(start_time, end_time, video_name), tuple(start_time, end_time, video_name)]
        """
        # 不是 storeinout/carinout 事件，直接计算 event_start_time, event_end_time 作为window
        if event.Type not in self.events_just_focus_head_tail:
            return [self._get_actual_start_end_time(event) + [event.video_name]]
        
        windows = []
        for i, time in enumerate([event.StartTime, event.EndTime]):
            # 根据ts获取对应的video_slice_name
            best_channel = event_label.get_best_channel_by_ts(event.Pid, string_to_ts(time))
            if best_channel is None:
                logging.warning(f"Can't find best coverage camera for {['start', 'end'][i]}:{time} of event:{event}, skipping it.")
                windows.append((0, 0, None))  # 跳过该事件，返回空列表
                continue
            
            video_name = self.get_video_name_by_ts(event, time, best_channel)
            
            # 获取video的start/end time seconds
            video_start_seconds, video_end_seconds = self._get_video_start_end_seconds(event.date, video_name)

            # 将事件开始和结束时间转换为秒
            event_time_sec = time.hour * 3600 + time.minute * 60 + time.second

            # 计算window的开始和结束时间（秒）
            window_start_time = max(video_start_seconds, event_time_sec - offset)
            window_end_time = min(video_end_seconds, event_time_sec + offset)

            if window_end_time <= window_start_time:
                windows.append((0, 0, None))  # 跳过该事件，返回空列表
                continue
            
            # 计算相对于video_start_time的偏移秒数作为实际的剪辑时间
            actual_start_time = window_start_time - video_start_seconds
            actual_end_time = window_end_time - video_start_seconds
            windows.append((actual_start_time, actual_end_time, video_name))

        return windows
    
    @staticmethod
    def _cutting(video_path, output_file, video_local_name, actual_start_time, actual_end_time):
        # 使用moviepy剪辑视频
        try:
            # 直接使用VideoFileClip和subclip方法来剪辑视频
            with VideoFileClip(str(video_path)) as video:
                new_video = video.subclip(actual_start_time, actual_end_time)
                new_video.write_videofile(str(output_file), codec="libx264", audio_codec="aac")
            
            logging.info(f"Video {video_local_name} cut successfully.")
        except Exception as e:
            logging.info(f"Error cutting video {video_local_name}: {e}")

    def cut_video(self, event_label, event, window_offset=5):
        """ 将本地的15分钟一段的视频，按照给定的event时间进行切片
        Args:
            event_label: first event label of this bmk, it not contains all events time range!!!
            event: event object.
        Returns:
            None.
        """
        assert isinstance(event, BaseEvent), "event must be instacne of subclass of BaseEvent"

        if event.Type not in self.events_just_focus_head_tail:
            return  # 不是 storeinout/carinout 事件，直接返回，不需要切片

        video_local_name = event.get_local_video_name()
        start_time = event.StartTime.strftime("%H:%M:%S")  # 确保时间格式是HH:MM:SS
        end_time = event.EndTime.strftime("%H:%M:%S")

        # 获取实际剪辑的开始和结束时间（秒）
        windows = self.get_actual_start_end_window_time(event_label, event, window_offset)

        # 设置路径
        repo_root = Path(__file__).parents[3]
        output_path = repo_root / self.output_path
        output_path.mkdir(parents=True, exist_ok=True)  # 创建输出目录
        output_file = output_path / f"{video_local_name}.mp4"  # 假设输出文件为mp4格式

        for idx, window in enumerate(windows):
            if window is None or window[-1] is None:
                continue

            (actual_start_time, actual_end_time, video_name) = window

            # 这里每个window可能对应不同的video，所以需要根据video_name来获取video_path
            video_path = repo_root / self.get_video_path(event.video_name) / video_name

            logging.info(f"Window: {start_time}~{end_time} -> {ts_to_string(actual_start_time)}~{ts_to_string(actual_end_time)}")
            # 如果是inout事件，我们只需要剪辑开头和结尾的window，所以会有多个视频，需要给clips编号
            if event.Type in self.events_just_focus_head_tail:
                output_file = output_path / f"{video_local_name}_{['IN', 'OUT'][idx]}_{getattr(event, 'RegionType', 'Store')}.mp4"
            
            # Excute cutting
            self._cutting(video_path, output_file, video_local_name, actual_start_time, actual_end_time)


