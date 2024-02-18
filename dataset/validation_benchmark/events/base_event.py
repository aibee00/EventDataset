from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class BaseEvent:
    Pid: Optional[str] = field(default="")
    StaffID: Optional[str] = field(default="")
    StartTime: Optional[str] = field(default="")
    EndTime: Optional[str] = field(default="")
    Type: Optional[str] = field(default="")
    issue_category: Optional[str] = field(default="")
    issue_details: Optional[str] = field(default="")
    record: Optional[str] = field(default="")

    site_id: str = ""  # 站点ID
    date: str = ""  # 日期
    video_name: str = ""  # 视频名称
    channel_id: str = ""  # 摄像头channel
    task_id: str = ""  # 任务ID

    def __post_init__(self):
        # 将 StartTime/EndTime 字符串转换为 datetime.time 对象
        self.StartTime = datetime.strptime(self.StartTime, '%H:%M:%S').time()
        self.EndTime = datetime.strptime(self.EndTime, '%H:%M:%S').time()

        # 设置默认的 video_name
        self.video_name = self.get_local_video_name()

    @staticmethod
    def time2str(t: datetime.time):
        return t.strftime('%H%M%S')  # 转换为字符串格式
    
    def get_local_video_name(self,):
        """ 获取本地视频名称
        """
        st = self.time2str(self.StartTime)
        et = self.time2str(self.EndTime)

        if self.Type == "INDIVIDUAL_RECEPTION":
            video_name = f"{self.Pid}-{self.StaffID}-{self.Type}-{st}-{et}"
        else:
            video_name = f"{self.Pid}-{self.Type}-{st}-{et}"  # 设置默认的 video_name
        return video_name
    
    def set_properties(self, **properties):
        """ 更新多个属性值

        Args:
            properties: 一个或多个键值对，键是属性名，值是相应的新值。
        """
        for property_name, value in properties.items():
            setattr(self, property_name, value)