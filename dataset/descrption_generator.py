import os
import os.path as osp
import shutil
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from common import STORE_INOUT_NEAR_TIME, cv2AddChineseText, merge_bboxes_at_one_channel, logger
from dataset.prompt_encoder import PromptEncoder
from get_tracks import TrackLoader
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
        self.context = set()
        self.index = 0

    def __str__(self) -> str:
        """ 生成label prompt的格式

        prompt格式：{'img': img_path, 'annotation': desc, 'context': context}
        """
        return f"{self.prompt}"
    
    @property
    def prompt(self, ) -> dict:
        """ 生成label prompt的样式
        """
        return {'img': self.__path, 'annotation': ";".join(self.desc), 'context': ";".join(self.context)}

    @property
    def path(self):
        return self.__path
    
    @path.setter
    def path(self, new_path):
        self.__path = new_path


class MyDict(dict):
    def __str__(self) -> str:
        string = "{"
        for k, v in self.items():
            k = str(k)
            v = str(v)
            string += f"{k}:{v},"
        string += "}"
        return string

    def value2str(self,) -> dict:
        for k, v in self.items():
            if isinstance(v, ImageObject):
                self[k] = v.prompt
            else:
                self[k] = str(v)
        return self
        

class ImageDescriptor(object):
    def __init__(self, root_path):
        self.imgs_path = osp.join(root_path, "imgs")
        if not osp.exists(self.imgs_path):
            os.makedirs(self.imgs_path)
        
        # Table to record img object
        self.img_table = MyDict()
        self.__img_num = 0

    @property
    def img_names(self):
        return self.img_table.keys()

    @property
    def total_num(self):
        return self.__img_num
    
    def register(self, org_img_path, img_name):
        """ 把img_object注册到img_table中
        """
        # 保存图片
        img_path = osp.join(self.imgs_path, img_name)
        if not osp.exists(img_path):
            # cv2保存图片
            logger.info(f"Saving img : {img_name} to {self.imgs_path}...")
            # 从org_img_path拷贝图片到img_path
            shutil.copy(org_img_path, img_path)
        
        if img_name not in self.img_table:
            img_object = ImageObject(img_path)

            # update img index
            img_object.index = self.__img_num
            self.__img_num += 1
        
            # update img_table
            self.img_table[img_name] = img_object

    def add_description(self, img_path, description, context=None):
        """ 向图片index添加描述
        """
        if not os.path.exists(img_path):
            raise ValueError(f"Image {img_path} not exists")
        
        img_name = osp.basename(img_path)
        if img_name not in self.img_table:
            logger.error(f"Image {img_name} not in img_table")
            return 
        
        # Update self.img_table中的img_object的描述属性
        img_object = self.img_table.get(img_name)
        img_object.desc.add(description)

        # Add context to img_object
        if context is not None and context != "" and context not in img_object.context:
            img_object.context.add(context)

    def update_dataset_path(self, new_dataset_path):
        """ 替换img_table中的img_object的path层空间
        """
        for img_name, img_object in self.img_table.items():
            img_object.path = osp.join(new_dataset_path, "imgs", img_name)


########################################## Begin Template Class Define ####################################
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
    

class ContextTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        bbox = None
        object_name = "Person"

        if self.event['event_type'] == "COMPANION":
            template_merge = ""
            pids = self.event['pids']
            for pid in pids:
                pid_bboxes = self.event.get('pid_bboxes', None)
                if pid_bboxes:
                    bbox = pid_bboxes.get(pid, None)
                    template_merge += template.format(object_name, pid, bbox)
            template = template_merge

        else:
            pid_bboxes = self.event.get('pid_bboxes', None)
            if pid_bboxes:
                bbox = pid_bboxes.get(self.event['pid'], None)
            template = template.format(object_name, self.event['pid'], bbox)
        
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

    def _convert_pid_to_bbox_embedding(self, pid, ts, channel):
        """ Deprecated
        Convert pid to its bbox_embedding with position encoding
        
        Args:
             pid: str
             ts: int
             channel: str
        
        Returns:
             embedding: List([4])

        Note:
            这个函数是计算每个ts的bbox的位置编码，速速太慢；且因为外部循环先迭代ts再迭代pid，所以存在计算冗余
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
    
    def _get_pid_bbox_embedding(self, pid_bbox_embeddings, pid, ts, ch):
        """ 获取某个pid的所有有效时刻所有channel的bboxes，并进行位置编码

        Args:
            pid: str
            ts: int
            ch: str
        
        Returns:
            bbox_embedding: List()
        """
        bbox_embeding = []
        if pid not in pid_bbox_embeddings:
            return bbox_embeding
        
        if ch not in pid_bbox_embeddings[pid]:
            return bbox_embeding
        
        if ts not in pid_bbox_embeddings[pid][ch]:
            return bbox_embeding
        
        return pid_bbox_embeddings[pid][ch][ts]

    def generate_prompt(self, event, pid_bbox_embeddings, channel, ts=None) -> str:
        event_embed = deepcopy(event)

        # 将pid用bbox_embedding表示
        if event["event_type"] == "COMPANION":
            pids = []
            for pid in event["pids"]:
                bbox_embed = self._get_pid_bbox_embedding(pid_bbox_embeddings, pid, ts, channel)

                if len(bbox_embed) == 0:
                    continue

                pids.append(f"<{str(bbox_embed)}>")
            
            if not pids:
                return ""
            
            event_embed["pids"] = pids
        else:
            bbox_embed = self._get_pid_bbox_embedding(pid_bbox_embeddings, event["pid"], ts, channel)
            
            if len(bbox_embed) == 0:
                return ""
            
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


class PromptDescriptorV2(PromptDescriptor):

    """ V2 版本说明
    
    这个版本主要是实现:
        - 将pid的名字替换为该pid的body_patch的box归一化后的表示，
        - box归一化是指坐标占比(based on img.shape)
    """

    def __init__(self, description_file_path, area_descriptor, pid_output_path):
        super().__init__(description_file_path, area_descriptor)

        self.track_loader = TrackLoader(pid_output_path)


    def _normalize_pid_bboxes(self, w, h, pid, ts, channel):
        """ 
        Normalize all bboxes of pid
        
        Args:
             pid: str
             ts: int
             channel: str
        
        Returns:
             embedding: List([4])
        """
        def get_norm_box(bbox, w, h):
            """ 
            Normalize a bbox
            
            Args:
                bbox: List([4])
                w: int
                h: int
            
            Returns:
                embedding: List([4])
            """
            x1, y1, x2, y2 = bbox
            x1 = x1 / w
            y1 = y1 / h
            x2 = x2 / w
            y2 = y2 / h
            return [x1, y1, x2, y2]

        # 根据pid和ts获取带有channel的boxes集合
        bboxes = self.track_loader.get_bboxes(pid, ts)

        # 把这一秒的所有bboxes进行合并by average, 返回{ch: BoundingBox}
        bboxes = merge_bboxes_at_one_channel(bboxes, channel)

        if channel not in bboxes:
            return []

        bbox = bboxes[channel].xyxy

        # 将BoundingBox进行归一化
        bbox_norm = get_norm_box(bbox, w, h)

        return bbox_norm

    def generate_prompt(self, event, ts=None) -> str:

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
        
    def generate_context(self, event, ts, channel, img_shape):
        """ 生成背景描述

        Args:
            event: Dict
            ts: int
            channel: str
            img_shape: List([3])
        
        Returns:
            context: str
        """

        h, w, _ = img_shape

        # 将pid用bbox_embedding表示
        if event["event_type"] == "COMPANION":
            for pid in event["pids"]:
                bbox_norm = self._normalize_pid_bboxes(w, h, pid, ts, channel)

                if len(bbox_norm) == 0:
                    continue

                bbox_norm = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm)
                event.setdefault("pid_bboxes", {})[pid] = bbox_norm

        else:
            bbox_norm = self._normalize_pid_bboxes(w, h, event["pid"], ts, channel)
            
            if len(bbox_norm) == 0:
                return ""
            
            bbox_norm = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm)
            event.setdefault("pid_bboxes", {})[event["pid"]] = bbox_norm

        txt_file = f"{self.description_file_path}/context.txt"
        return str(ContextTemplate(txt_file, event, self.area_descriptor))
