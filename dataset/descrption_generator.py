import os
import os.path as osp
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from common import STORE_INOUT_NEAR_TIME, cv2AddChineseText, merge_bboxes_at_one_channel
from dataset.prompt_encoder import PromptEncoder
from get_tracks import TrackLoader
import torch
from copy import deepcopy
import yaml


class ImageObject(object):

    def __init__(self, img_path):
        self.__path = img_path
        self.name = osp.basename(img_path).split('.')[0]
        self.img = None
        self.img_size = None
        self.img_shape = None
        self.desc = set()
        self.index = 0

    def __str__(self) -> str:
        """ 生成label prompt的格式

        prompt格式：{'img': img_path, 'annotation': desc}
        """
        return f"{self.prompt}"
    
    @property
    def prompt(self, ) -> dict:
        """ 生成label prompt的样式
        """
        return {'img': self.__path, 'annotation': ";".join(self.desc)}

    @property
    def path(self):
        return self.__path
    
    @path.setter
    def path(self, new_path):
        self.__path = new_path


# 带参数的装饰器
def wrapper_str(func):
    def wrapper(self, *args, **kwargs):
        assert osp.exists(self.template_file), f"Template file not found in: {self.template_file}"
        result = func(self, *args, **kwargs)
        return result
    return wrapper


class Template(metaclass=ABCMeta):
    template_file = None

    def __init__(self, template_file, event, area_descriptor=None, ts=None):
        self.template_file = template_file
        self.event = event
        self.area_descriptor = area_descriptor
        self.ts = ts
    
    @abstractmethod
    def __str__(self):
        pass


class StoreInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None, ts=None):
        super().__init__(template_file, event, area_descriptor, ts)

    # 添加装饰器
    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()

        # 如果self.ts不为None, 则根据ts来判断是进店还是出店
        inout = "经过"
        if self.ts:
            # ts 在开始时间附近(前后10s)则标记inout为'进入'，结束时间附近标记为'走出'
            if abs(self.ts - self.event['start_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "进入"
            elif abs(self.ts - self.event['end_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "走出"
        
        template = template.format(self.event['pid'], inout)
        
        return template
    

class RegionVisitTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        region = self.area_descriptor.car_region[self.event['region_id']]
        reg_name = region.name if region.name else region.type
        template = template.format(self.event['pid'], reg_name)
        
        return template
    

class RegionInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None, ts=None):
        super().__init__(template_file, event, area_descriptor, ts)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        # 如果self.ts不为None, 则根据ts来判断是进店还是出店
        inout = "内部访问"
        if self.ts:
            # ts 在开始时间附近(前后10s)则标记inout为'进入'，结束时间附近标记为'走出'
            if abs(self.ts - self.event['start_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "进入"
            elif abs(self.ts - self.event['end_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "走出"
        
        region = self.area_descriptor.internal_region[self.event['region_id']]
        reg_name = region.name if region.name else region.type
        template = template.format(self.event['pid'], inout, reg_name)
        
        return template
    

class CarInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None, ts=None):
        super().__init__(template_file, event, area_descriptor, ts)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        # 如果self.ts不为None, 则根据ts来判断是进店还是出店
        inout = "内部访问"
        if self.ts:
            # ts 在开始时间附近(前后10s)则标记inout为'进入'，结束时间附近标记为'走出'
            if abs(self.ts - self.event['start_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "进入"
            elif abs(self.ts - self.event['end_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "走出"
        
        region = self.area_descriptor.car_region[self.event['region_id']]
        reg_name = region.name if region.name else self.event['region_id']
        template = template.format(self.event['pid'], inout, reg_name)
        
        return template
    

class CompanionTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        template = template.format(self.event['pids'])
        
        return template
    

class IndividualReceptionTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        template = template.format(self.event['staff_id'], self.event['pid'])
        
        return template   


class PromptDescriptor(object):

    def __init__(self, description_file_path, area_descriptor):
        self.description_file_path = description_file_path
        self.area_descriptor = area_descriptor

    def parse_annotation(self, annotation):
        """ parse annotation into prompt
        """
        if isinstance(annotation, ImageObject):
            return str(annotation.prompt)
        return annotation["annotation"]

    def draw_prompt(self, img, prompt):
        """ 将文本prompt和img同框显示
        :param img: cv2读取的image对象
        :param prompt:
        :return: new img with prompt
        """
        # 将文本prompt和img同框显示,上半部分显示img图片，下半部分显示文本prompt，
        # 如框内文字不充满，则下方补空样式,最后把上下两部分合并为一张大图返回
        
        # 初始化一个和img同样大小的空白数据, 用白色填充
        img_blank = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        img_blank.fill(255)

        # 设置文本prompt的字位，大小，颜艬
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        # 计算文本prompt的定位
        text_size, baseline = cv2.getTextSize(prompt, font, font_scale, thickness)
        text_w, text_h = text_size
        # 计算文本prompt的中心定位
        # text_x = (img.shape[1] - text_w) // 2
        # text_y = (img.shape[0] + text_h) // 2
        text_x = 20
        text_y = 20

        # cv2.putText(img_blank, prompt, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        img_blank = cv2AddChineseText(img_blank, prompt, (text_x, text_y), color, textSize=60)

        # 将img和img_blank合将为一大图
        img_blank = np.concatenate((img, img_blank), axis=0)
        return img_blank


    def generate_prompt(self, event, channel, ts=None):
        if event["event_type"] == "STORE_INOUT":
            txt_file = f"{self.description_file_path}/store_inout.txt"
            return str(StoreInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_VISIT":
            txt_file = f"{self.description_file_path}/region_visit.txt"
            return str(RegionVisitTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            txt_file = f"{self.description_file_path}/region_inout.txt"
            return str(RegionInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            txt_file = f"{self.description_file_path}/car_inout.txt"
            return str(CarInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "COMPANION":
            txt_file = f"{self.description_file_path}/companion.txt"
            return str(CompanionTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            txt_file = f"{self.description_file_path}/individual_reception.txt"
            return str(IndividualReceptionTemplate(txt_file, event, self.area_descriptor))
        else:
            raise Exception(f"Event type {event['event_type']} not supported.")
        
    
class PromptDescriptorV1(PromptDescriptor):

    """ V1 版本说明
    
    这个版本主要是实现:
        - 将pid的名字替换为该pid的body_patch的位置编码嵌入表示
        - 采用问答的形式，将问题作为Q-former的Text输入，答案作为label与输出就损失
        - 加入一些中间信息，例如，加入朝向信息
    """

    def __init__(self, description_file_path, area_descriptor, pid_output_path):
        super().__init__(description_file_path, area_descriptor)
        self.track_loader = TrackLoader(pid_output_path)

        # 加载encoder_config
        config_path = osp.join(osp.dirname(__file__), 'config.yaml')
        assert osp.exists(config_path), f"config.yaml not found in {osp.dirname(__file__)}"
        with open(config_path, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader).get('PromptEncoder', {})
        print(f"Loaded PromptEncoder Config: {cfg}")
        self.prompt_encoder = PromptEncoder(
            embed_dim=cfg.get('embed_dim', 64),
            image_embedding_size=cfg.get('image_embedding_size', (32,32)),
            input_image_size=cfg.get('input_image_size', (32,32)),
            mask_in_chans=cfg.get('mask_in_chans', 1)
        )

    def convert_pid_to_bbox_embedding(self, pid, ts, channel):
        """ Convert pid to its bbox_embedding with position encoding
        Args:
             pid: str
             ts: int
             channel: str
        Returns:
             embedding: List([4])
        """
        # 根据pid和ts获取带有channel的boxes集合
        bboxes = self.track_loader.get_bboxes(pid, ts)

        # 把这一秒的所有bboxes进行合并by average, 返回{ch: BoundingBox}
        bboxes = merge_bboxes_at_one_channel(bboxes, channel)

        if channel not in bboxes:
            return []

        bbox = bboxes[channel]

        # 将bboxes中的BoundingBox转为四个角点坐标的形式并转为tensor
        bbox = bbox.convert_coords_to_4_corners_points().get_coords_as_tensor()  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

        # 将bbox(shape(4，2))进行处理，将其进行padding，将其进行reshape(1，4，2）
        bbox = bbox.unsqueeze(0)

        # 将BoundingBox进行位置编码
        bbox_embed = self.prompt_encoder.get_box_embed(bbox)
        bbox_embed = bbox_embed.squeeze()  # shape(4)

        # 将tensor转为list
        bbox_embed = bbox_embed.tolist()

        return bbox_embed

    def generate_prompt(self, event, channel, ts=None):
        event_embed = deepcopy(event)

        if not event_embed:
            return ""

        # 将pid用bbox_embedding表示
        if event["event_type"] == "COMPANION":
            pids = []
            for pid in event["pids"]:
                bbox_embed = self.convert_pid_to_bbox_embedding(pid, ts, channel)
                pids.append(f"<{str(bbox_embed)}>")
            event_embed["pids"] = pids
        else:
            bbox_embed = self.convert_bbox_to_embedding(event["pid"], ts, channel)
            event_embed["pid"] = f"<{str(bbox_embed)}>"

        if event["event_type"] == "STORE_INOUT":
            txt_file = f"{self.description_file_path}/store_inout.txt"
            return str(StoreInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_VISIT":
            txt_file = f"{self.description_file_path}/region_visit.txt"
            return str(RegionVisitTemplate(txt_file, event_embed, self.area_descriptor))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            txt_file = f"{self.description_file_path}/region_inout.txt"
            return str(RegionInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            txt_file = f"{self.description_file_path}/car_inout.txt"
            return str(CarInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "COMPANION":
            txt_file = f"{self.description_file_path}/companion.txt"
            return str(CompanionTemplate(txt_file, event_embed, self.area_descriptor))
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            txt_file = f"{self.description_file_path}/individual_reception.txt"
            return str(IndividualReceptionTemplate(txt_file, event_embed, self.area_descriptor))
        else:
            raise Exception(f"Event type {event['event_type']} not supported.")

