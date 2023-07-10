import os
import os.path as osp
import shutil
import cv2
import numpy as np
from collections import UserDict
from common import cv2AddChineseText, merge_bboxes_at_one_channel, logger
from dataset.prompt_encoder import PromptEncoder
from get_tracks import TrackLoader
from copy import deepcopy
import yaml

from template import *


class ImageObject(object):

    def __init__(self, img_path):
        self.__path = img_path
        self.name = osp.basename(img_path).split('.')[0]
        self.img = None
        self.img_size = None
        self.img_shape = None
        self.desc = set()
        self.context = set()
        self.bbox_embedding = set()
        self.index = 0

        self.dynamic_attribute_names = set()  # record new added attrbutes

        # context to pid_index
        self.pid_to_index = dict()

        # region_name to region index
        self.region_to_index = dict()

    def __str__(self) -> str:
        """ 生成label prompt的格式

        prompt格式：{'img': img_path, 'annotation': desc, 'context': context, ...}
        """
        return f"{self.prompt}"
    
    @property
    def prompt(self, ) -> dict:
        """ 生成label prompt的样式
        """
        # replace pid with index
        self.replace_pid_with_index()
        
        prompt = {'img': self.path, 
                  'annotation': ";".join(self.desc), 
                  'context': ";".join(sorted(self.context)),
                  'bbox_embedding': ";".join(sorted(self.bbox_embedding)),
                  'pid_to_index': self.pid_to_index}
        for attr_name in self.dynamic_attribute_names:
            # attr_name: {regiontype}_context
            prompt[attr_name] = ";".join(sorted(getattr(self, attr_name)))

            attr_prefix = attr_name.split("_")[0]  # region type
            if hasattr(self, attr_prefix + "_to_index"):
                prompt[attr_name + "_to_index"] = getattr(self, attr_prefix + "_to_index")
        return prompt
    
    @property
    def path(self):
        return self.__path
    
    @path.setter
    def path(self, new_path):
        self.__path = new_path

    def register(self, key: str, value: set = set()):
        """ Register new attribute into dynamic_attributes if self did not has it
        """
        if not hasattr(self, key):
            setattr(self, key, value)
            self.dynamic_attribute_names.add(key)
            # print(f"Register Success! New attributes list: {self.dynamic_attribute_names}")
        else:
            print(f"{key} already exists in {self.__dict__}")

    def replace_pid_with_index(self, ):
        """ Replace pid with index in description
        """
        desc_list = list(self.desc)
        for i, desc in enumerate(desc_list):
            for pid, index in self.pid_to_index.items():
                if pid not in desc:
                    continue

                desc_list[i] = desc_list[i].replace(f"<{pid}>", "Person{}".format(index))
        self.desc = set(desc_list)

        # replace pid with index in context
        cont_list = list(self.context)
        for i, cont in enumerate(cont_list):
            for pid, index in self.pid_to_index.items():
                if pid not in cont:
                    continue

                cont_list[i] = cont_list[i].replace(f"<{pid}>", "{}".format(index))
        self.context = set(cont_list)

        # replace pid with index in bbox_embedding
        emb_list = list(self.bbox_embedding)
        for i, emb in enumerate(emb_list):
            for pid, index in self.pid_to_index.items():
                if pid not in emb:
                    continue

                emb_list[i] = emb_list[i].replace(f"<{pid}>", "{}".format(index))
        self.bbox_embedding = set(emb_list)

        # Replace region_name with index in description
        for attr_name in self.dynamic_attribute_names:
            region_type = attr_name.split('_')[0]  # {regiontype}_context

            # 替换self.{regiontype}_context中的region_name为{regiontype}{index}
            cont_list = list(getattr(self, attr_name))
            for i, cont in enumerate(cont_list):
                for region_name, index in getattr(self, f"{region_type}_to_index").items():
                    if region_name not in cont:
                        continue

                    cont_list[i] = cont_list[i].replace(f"<{region_name}>", f"{region_type}{index}")
            setattr(self, attr_name, set(cont_list))

            # 替换annotation中的RegionName为f'{regiontype}{index}'
            desc_list = list(self.desc)
            for i, desc in enumerate(desc_list):
                for region_name, index in getattr(self, f"{region_type}_to_index").items():
                    if region_name not in desc:
                        continue

                    desc_list[i] = desc_list[i].replace(f"<{region_name}>", f"{region_type}{index}")
            self.desc = set(desc_list)


class MyDict(UserDict):
    def __str__(self) -> str:
        string = "{"
        for k, v in self.items():
            k = str(k)
            v = str(v)
            string += f"{k}:{v},"
        string += "}"
        return string

    def value2str(self,) -> dict:
        ret = {}
        for k, v in self.items():
            if isinstance(v, ImageObject):
                ret[k] = v.prompt
            else:
                ret[k] = str(v)
        return ret
        

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
    
    def get_img_object(self, img_name):
        """ 获取img_object
        """
        if img_name not in self.img_table:
            logger.error(f"Image {img_name} not in img_table")
            return None
        
        return self.img_table.get(img_name)
    
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

    def add_description(self, img_path, description, other_attributes={}):
        """ 向图片index添加描述
        """
        SEP = ";"

        def get_pid_from_description(desc):
            """ desc格式: <pid_name>:[x1,y1,x2,y2]
            """
            pid = ""
            for part in desc.split("<"):
                if '>' in part:
                    pid = part.split(">")[0]
                    break
            return pid
        
        def update_index_map(img_object, name, attribute, pid_to_index):
            if SEP in attribute:
                for desc in attribute.split(SEP):
                    pid = get_pid_from_description(desc)
                    if pid and pid not in pid_to_index:
                        # update new pid and its desc
                        pid_to_index[pid] = len(pid_to_index)
                        # 更新描述到属性中
                        getattr(img_object, name).add(desc)
            else:
                pid = get_pid_from_description(attribute)
                if pid and pid not in pid_to_index:
                    # update new pid and its desc
                    pid_to_index[pid] = len(pid_to_index)
                    # 更新描述到属性中
                    getattr(img_object, name).add(attribute)

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
        if other_attributes:
            for name, attribute in other_attributes.items():
                if not hasattr(img_object, name):
                    img_object.register(name, set())

                # 检查attribute对应的pid是否已经存在
                # Note: 一般情况下desc是一个描述，
                # 但group事件中有多个pid所以会有多个描述的情况，以分号(;)分隔

                if name == 'context':  # 默认是保存pid的context
                    pid_to_index = img_object.pid_to_index
                else:
                    region_type = name.split('_')[0]
                    if not hasattr(img_object, f"{region_type}_to_index"):
                        setattr(img_object, f"{region_type}_to_index", dict())
                    
                    pid_to_index = getattr(img_object, f"{region_type}_to_index")

                update_index_map(img_object, name, attribute, pid_to_index)


    def update_dataset_path(self, new_dataset_path):
        """ 替换img_table中的img_object的path层空间
        """
        for img_name, img_object in self.img_table.items():
            img_object.path = osp.join(new_dataset_path, "imgs", img_name)


class PromptDescriptor(object):

    def __init__(self, template_file_path, area_descriptor):
        self.template_file_path = template_file_path
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
            txt_file = f"{self.template_file_path}/store_inout.txt"
            return str(StoreInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_VISIT":
            txt_file = f"{self.template_file_path}/region_visit.txt"
            return str(RegionVisitTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            txt_file = f"{self.template_file_path}/region_inout.txt"
            return str(RegionInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            txt_file = f"{self.template_file_path}/car_inout.txt"
            return str(CarInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "COMPANION":
            txt_file = f"{self.template_file_path}/companion.txt"
            return str(CompanionTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            txt_file = f"{self.template_file_path}/individual_reception.txt"
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

    def __init__(self, template_file_path, area_descriptor, pid_output_path):
        super().__init__(template_file_path, area_descriptor)
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
            txt_file = f"{self.template_file_path}/store_inout.txt"
            return str(StoreInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_VISIT":
            txt_file = f"{self.template_file_path}/region_visit.txt"
            return str(RegionVisitTemplate(txt_file, event_embed, self.area_descriptor))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            txt_file = f"{self.template_file_path}/region_inout.txt"
            return str(RegionInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            txt_file = f"{self.template_file_path}/car_inout.txt"
            return str(CarInoutTemplate(txt_file, event_embed, self.area_descriptor, ts))
        elif event["event_type"] == "COMPANION":
            txt_file = f"{self.template_file_path}/companion.txt"
            return str(CompanionTemplate(txt_file, event_embed, self.area_descriptor))
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            txt_file = f"{self.template_file_path}/individual_reception.txt"
            return str(IndividualReceptionTemplate(txt_file, event_embed, self.area_descriptor))
        else:
            raise Exception(f"Event type {event['event_type']} not supported.")


class PromptDescriptorV2(PromptDescriptor):

    """ V2 版本说明
    
    这个版本主要是实现:
        - 将pid的名字替换为该pid的body_patch的box归一化后的表示，
        - box归一化是指坐标占比(based on img.shape)
    """

    def __init__(self, template_file_path, area_descriptor, pid_output_path):
        super().__init__(template_file_path, area_descriptor)

        self.track_loader = TrackLoader(pid_output_path)

    @staticmethod
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
    
    @staticmethod
    def get_norm_points(points, w, h):
        """ 
        Normalize a points
        
        Args:
            points: List(point1, point2,...), shape(n, 2)
            w: int
            h: int
        
        Returns:
            embedding: List(point1, point2,...), shape(n, 2)
        """
        points = np.array(points)
        points[:, 0] = points[:, 0] / w
        points[:, 1] = points[:, 1] / h
        return points.tolist()

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
        # 根据pid和ts获取带有channel的boxes集合
        bboxes = self.track_loader.get_bboxes(pid, ts)

        # 把这一秒的所有bboxes进行合并by average, 返回{ch: BoundingBox}
        bboxes = merge_bboxes_at_one_channel(bboxes, channel)

        if channel not in bboxes:
            return []

        bbox = bboxes[channel].xyxy

        # 将BoundingBox进行归一化
        bbox_norm = self.get_norm_box(bbox, w, h)

        return bbox_norm

    def generate_prompt(self, event, ts=None) -> str:

        if event["event_type"] == "STORE_INOUT":
            txt_file = f"{self.template_file_path}/store_inout.txt"
            return str(StoreInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_VISIT":
            txt_file = f"{self.template_file_path}/region_visit.txt"
            return str(RegionVisitTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            txt_file = f"{self.template_file_path}/region_inout.txt"
            return str(RegionInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            txt_file = f"{self.template_file_path}/car_inout.txt"
            return str(CarInoutTemplate(txt_file, event, self.area_descriptor, ts))
        elif event["event_type"] == "COMPANION":
            txt_file = f"{self.template_file_path}/companion.txt"
            return str(CompanionTemplate(txt_file, event, self.area_descriptor))
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            txt_file = f"{self.template_file_path}/individual_reception.txt"
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
                    event.setdefault("pid_bboxes", {})[pid] = bbox_norm
                    continue

                bbox_norm = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm)
                event.setdefault("pid_bboxes", {})[pid] = bbox_norm
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            pid = event["pid"]
            staff = event["staff_id"]
            bbox_norm_c = self._normalize_pid_bboxes(w, h, pid, ts, channel)
            bbox_norm_s = self._normalize_pid_bboxes(w, h, staff, ts, channel)
            
            try:
                bbox_norm_c = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm_c)
                bbox_norm_s = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm_s)
            except:  # bbox_norm_s is empty
                bbox_norm_c = bbox_norm_c
                bbox_norm_s = bbox_norm_s
            event.setdefault("pid_bboxes", {})[pid] = bbox_norm_c
            event.setdefault("pid_bboxes", {})[staff] = bbox_norm_s
        else:
            bbox_norm = self._normalize_pid_bboxes(w, h, event["pid"], ts, channel)
            
            if len(bbox_norm) == 0:
                event.setdefault("pid_bboxes", {})[event["pid"]] = bbox_norm
                return ""
            
            bbox_norm = "[%.2f, %.2f, %.2f, %.2f]" % tuple(bbox_norm)
            event.setdefault("pid_bboxes", {})[event["pid"]] = bbox_norm

        txt_file = f"{self.template_file_path}/context.txt"
        return str(ContextTemplate(txt_file, event, self.area_descriptor))
    
    def generate_region_context(self, event, ts, channel, img_shape):
        """ 生戯各业务包域的描述

        Args:
            event: Dict
            ts: int
            channel: str
        
        Returns:
            context: str
        """
        # 从event中获取region object
        if event["event_type"] == "STORE_INOUT":
            if abs(ts - event['start_time']) < STORE_INOUT_NEAR_TIME * 2:
                region = self.area_descriptor.get_region_by_type_and_id(event["region_type"], event["in_door_id"])
            elif abs(ts - event['end_time']) < STORE_INOUT_NEAR_TIME * 2:
                try:
                    region = self.area_descriptor.get_region_by_type_and_id(event["region_type"], event["out_door_id"])
                except:
                    logger.error(f"{event} region not found!")
                    return ""
            else:
                region = self.area_descriptor.get_region_by_type_and_id(event["region_type"], event["in_door_id"])
        else:
            try:
                region = self.area_descriptor.get_region_by_type_and_id(event["region_type"], event["region_id"])
            except:
                # logger.warning(f"Event {event} not support in region context.")
                return ""
                

        if region is None:
            return ""

        # 将region的坐标从floor映射到 camera 的坐标系
        reg_coords = region.coords
        reg_coords = self.area_descriptor.map_floor_to_camera(reg_coords, channel)
        # 归一化
        h, w, _ = img_shape
        reg_coords_norm = self.get_norm_points(reg_coords, w, h)
        region.coords = reg_coords_norm

        event["region"] = region

        txt_file = f"{self.template_file_path}/region_context.txt"
        return str(RegionContextTemplate(txt_file, event, self.area_descriptor))
    

class PromptDescriptorV3(PromptDescriptorV2):

    """ V3 版本进阶版
    
    这一版主要是合并box_norm和bbox_embedding表示, 作为context添加到label中
    """

    def __init__(self, template_file_path, area_descriptor, pid_output_path):
        super().__init__(template_file_path, area_descriptor, pid_output_path)

    def generate_prompt(self, event, ts=None) -> str:
        """ 继承PromptDescriptorV2的.generate_prompt函数
        """
        return super().generate_prompt(event, ts)
    
    def generate_context(self, event, ts, channel, img_shape):
        """ 继承PromptDescriptorV2的.generate_context函数
        """
        return super().generate_context(event, ts, channel, img_shape)
    
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

    def generate_bbox_embedding(self, event, pid_bbox_embeddings, channel, ts=None) -> str:
        """ 生成pid的boundingbox的位置编码embedding
        """
        # 生成bbox_embedding表示
        if event["event_type"] == "COMPANION":
            for pid in event["pids"]:
                bbox_embed = self._get_pid_bbox_embedding(pid_bbox_embeddings, pid, ts, channel)

                if len(bbox_embed) == 0:
                    continue

                event.setdefault("pid_bbox_embeddings", {})[pid] = str(bbox_embed)
            
        else:
            bbox_embed = self._get_pid_bbox_embedding(pid_bbox_embeddings, event["pid"], ts, channel)
            
            if len(bbox_embed) == 0:
                return ""
            
            event.setdefault("pid_bbox_embeddings", {})[event["pid"]] = str(bbox_embed)
        
        txt_file = f"{self.template_file_path}/bbox_embedding.txt"
        return str(BboxEmbeddingTemplate(txt_file, event, self.area_descriptor))


