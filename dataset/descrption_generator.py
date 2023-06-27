import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from abc import ABCMeta, abstractmethod


# 定义一个自动换行的函数，最大长度为max_length, 并将多行文本保存到一个列表中返回
def auto_wrap(text, max_length, textSize, sep=";"):
    lines = []
    line = ''
    lineLen = 0

    bar = text if sep != ";" else text.split(sep)
    for word in bar:
        # 判断word是不是字母
        wlen = len(word) // 2 if word.isalpha() else len(word)
        
        # 更新lineLen
        lineLen += wlen
        
        if lineLen * textSize > max_length - wlen * textSize:
            lines.append(line)
            line = ''
            lineLen = 0

        line += word + ''
    lines.append(line)
    return lines
    

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(osp.join(osp.dirname(__file__), "simsun.ttc"), textSize, encoding="utf-8")

    # 将文本prompt画到img_blank上, 并调根据剩余空间调整字体大小使之填满整个img_blank区域，自动换行
    # 当字体超过img的宽度时自动换行
    events = auto_wrap(text, img.size[0], textSize)
    
    for event in events:
        lines = auto_wrap(event, img.size[0], textSize, sep="")
        for line in lines:
            print(f"line: {line}")

            # 如果超出了img的高度，则img在底部concat一块高度为(textSize + 8)、宽度为img.size[0]的空白区域
            if position[1] + textSize + 8 > img.size[1]:
                img_blank = Image.new('RGB', (img.size[0], (textSize + 8)), (255, 255, 255))
                img = np.concatenate((img, img_blank), axis=0)
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
            
            draw.text(position, line, textColor, font=fontStyle)
            position = (position[0], position[1] + textSize + 8)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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


class Template(metaclass=ABCMeta):
    def __init__(self, template_file, event, area_descriptor=None):
        self.template_file = template_file
        self.event = event
        self.area_descriptor = area_descriptor
    
    @abstractmethod
    def __str__(self):
        pass


class StoreInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        template = template.format(self.event['pid'])
        
        return template
    

class RegionVisitTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        region = self.area_descriptor.car_region[self.event['region_id']]
        reg_name = region.name if region.name else region.type
        template = template.format(self.event['pid'], reg_name)
        
        return template
    

class RegionInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        region = self.area_descriptor.internal_region[self.event['region_id']]
        reg_name = region.name if region.name else region.type
        template = template.format(self.event['pid'], reg_name)
        
        return template
    

class CarInoutTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        region = self.area_descriptor.car_region[self.event['region_id']]
        reg_name = region.name if region.name else region.type
        template = template.format(self.event['pid'], reg_name)
        
        return template
    

class CompanionTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        template = template.format(self.event['pids'])
        
        return template
    

class IndividualReceptionTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    def __str__(self):
        if not osp.exists(self.template_file):
            return f"Template file not found in: {self.template_file}"
        
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


    def generate_prompt(self, event):
        if event["event_type"] == "STORE_INOUT":
            return StoreInoutTemplate(f"{self.description_file_path}/store_inout.txt", event, self.area_descriptor).__str__()
        elif event["event_type"] == "REGION_VISIT":
            return RegionVisitTemplate(f"{self.description_file_path}/region_visit.txt", event, self.area_descriptor).__str__()
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "INTERNAL_REGION":
            return RegionInoutTemplate(f"{self.description_file_path}/region_inout.txt", event, self.area_descriptor).__str__()
        elif event["event_type"] == "REGION_INOUT" and event["region_type"] == "CAR":
            return CarInoutTemplate(f"{self.description_file_path}/car_inout.txt", event, self.area_descriptor).__str__()
        elif event["event_type"] == "COMPANION":
            return CompanionTemplate(f"{self.description_file_path}/companion.txt", event, self.area_descriptor).__str__()
        elif event["event_type"] == "INDIVIDUAL_RECEPTION":
            return IndividualReceptionTemplate(f"{self.description_file_path}/individual_reception.txt", event, self.area_descriptor).__str__()
        else:
            raise Exception(f"Event type {event['event_type']} not supported.")
        
    