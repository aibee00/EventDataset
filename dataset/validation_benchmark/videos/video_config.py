import os
from datetime import datetime, timedelta

from dataset.validation_benchmark.events.events import BaseEvent


class VideoConfig:
    """视频配置
    一个配置对应一个video, 对应一个event, 对应一个video_meta_file, 对应一个video_path
    """
    def __init__(self, video_id, task_id, channel_id, video_name, start_time, end_time):

        self._default_bucket = "https://store-lixiang.oss-cn-hangzhou-internal.aliyuncs.com/vista/video/"

        self.video_id = video_id  # {site_id}_{date}
        self.task_id = task_id  # task_id 验证时的 task_id，不同的 task_id 下载不同的视频
        self.channel_id = channel_id  # 视频通道号，例如 ch01001
        self.video_name = video_name  # 视频名称
        self.video_start_time = start_time  # 视频开始时间 format: "%H:%M:%S"
        self.video_end_time = end_time  # 视频结束时间 format: "%H:%M:%S"

        self.video_url = self.get_video_url()  # 视频地址，例如 oss://store-lixiang/vista/video/LIXIANG/anshan/wxhxd/20230906/806/ch01001/
        self.video_duration = self._get_duration(start_time, end_time)  # 视频时长，format: "%H:%M:%S"

        # 视频本身相关属性
        self.video_type = "mp4"  # 视频类型
        self.video_resolution = (image_height:=1440, image_width:=2560)  # 分辨率
        self.video_fps = 12  # 帧率
        self.video_size = 0  # 视频大小，单位为MB

        # 视频保存路径
        self.video_path = None  # 视频本身文件夹路径
        self.video_meta_path = None  # 视频信息文件夹路径
        self.video_meta_file = None  # 视频信息文件名

    def to_dict(self):
        # 将实例的属性转换为字典
        return {attr: getattr(self, attr) for attr in self.__dict__}

    @staticmethod
    def _get_duration(start, end):
        """ Return duration by format datetime.time like this: "%H:%M:%S"
        Args:
            start: start time. datetime.time. 
            end: end time.  datetime.time.
        Returns:
            duration: duration.
        """
        if isinstance(start, str): # 如果是字符串，转换为 datetime.time 对象
            start = datetime.strptime(start, "%H:%M:%S").time()
        if isinstance(end, str):
            end = datetime.strptime(end, "%H:%M:%S").time()
        
        # 将 time 对象转换为当天的 datetime 对象
        today = datetime.now().date()
        start_dt = datetime.combine(today, start)
        end_dt = datetime.combine(today, end)

        # 计算持续时间
        duration = end_dt - start_dt
        if duration.total_seconds() < 0:
            # 如果结束时间小于开始时间，考虑跨日情况
            duration += timedelta(days=1)

        # 手动计算小时、分钟、秒
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        # 格式化字符串
        duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        return duration_str

    @property
    def default_bucket(self,):
        return self._default_bucket
    
    @default_bucket.setter
    def default_bucket(self, value):
        self._default_bucket = value

    def get_video_url(self, ):
        assert getattr(self, 'video_id') is not None, "video_id must be set"
        assert getattr(self, 'task_id') is not None, "task_id must be set"
        assert getattr(self, 'channel_id') is not None, "channel_id must be set"
        assert getattr(self, 'video_name') is not None, "video_name must be set"

        default_url = os.path.join(
            self.default_bucket, 
            self.video_id.replace("_", "/"), 
            str(self.task_id), 
            self.channel_id,
            self.video_name,
        )
        return default_url
    
    def set_save_path(self, save_path):
        """ Set save path for video and video_meta.
        Args:
            save_path: save path.
        Returns:
            None.
        """
        assert isinstance(save_path, str), "save_path must be str"

        self.video_path = os.path.join(save_path, self.video_id, "video")
        self.video_meta_path = os.path.join(save_path, self.video_id, "meta")
        self.video_meta_file = os.path.join(self.video_meta_path, f"{self.video_name}.json")

        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)
        if not os.path.exists(self.video_meta_path):
            os.makedirs(self.video_meta_path)

    @classmethod
    def from_event(cls, event: BaseEvent, save_path: str):
        """ Get video config from event.
        Args:
            event: event object.
        Returns:
            config: VideoConfig object.
        """
        assert isinstance(event, BaseEvent), "event must be instacne of subclass of BaseEvent"

        video_config = cls(
            video_id=f"{event.site_id}_{str(event.date)}",
            task_id=event.task_id,
            channel_id=event.channel_id,
            video_name=event.video_name,
            start_time=str(event.StartTime),
            end_time=str(event.EndTime),  # 视频结束时间 format: "%H:%M:%S"
        )

        video_config.set_save_path(save_path)  # 设置保存路径

        return video_config