import os.path as osp
from abc import ABCMeta, abstractmethod
from common import STORE_INOUT_NEAR_TIME



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

        # Door caption
        if str(self.event['in_door_id']) not in self.area_descriptor.door_region and str(self.event['out_door_id']) not in self.area_descriptor.door_region:
           return ""
        door_in = self.area_descriptor.door_region[str(self.event['in_door_id'])] if str(self.event['in_door_id']) in  self.area_descriptor.door_region else None
        door_out = self.area_descriptor.door_region[str(self.event['out_door_id'])] if str(self.event['out_door_id']) in self.area_descriptor.door_region else None
        in_door_name = "Door" + door_in.id if door_in else "Door"
        out_door_name = "Door" + door_out.id if door_out else "Door"

        # 如果self.ts不为None, 则根据ts来判断是进店还是出店
        inout = "经过"
        door = in_door_name
        if self.ts:
            # ts 在开始时间附近(前后10s)则标记inout为'进入'，结束时间附近标记为'走出'
            if door_in and abs(self.ts - self.event['start_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "进入"
                door = in_door_name
            elif door_out and abs(self.ts - self.event['end_time']) < STORE_INOUT_NEAR_TIME * 2:
                inout = "走出"
                door = out_door_name
        
        template = template.format(self.event['pid'], inout, door)
        
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

        reg_name = "Car" + str(region.id)
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

        reg_name = "InternalRegion" + str(region.id)
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

        reg_name = "Car" + str(region.id)
        template = template.format(self.event['pid'], inout, reg_name)
        
        return template
    

class CompanionTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        pids = [f"<{pid}>" for pid in self.event['pids']]
        template = template.format(str(pids))
        
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
            template_merge = []
            pids = self.event['pids']
            for pid in pids:
                pid_bboxes = self.event.get('pid_bboxes', None)
                if pid_bboxes:
                    bbox = pid_bboxes.get(pid, None)
                    template_merge.append(template.format(object_name, pid, bbox))
            template = ";".join(template_merge)
        elif self.event['event_type'] == "INDIVIDUAL_RECEPTION":
            template_merge = []
            pids = [self.event['pid'], self.event['staff_id']]
            for pid in pids:
                pid_bboxes = self.event.get('pid_bboxes', None)
                if pid_bboxes:
                    bbox = pid_bboxes.get(pid, None)
                    template_merge.append(template.format(object_name, pid, bbox))
            template = ";".join(template_merge)
        else:
            pid_bboxes = self.event.get('pid_bboxes', None)
            if pid_bboxes:
                bbox = pid_bboxes.get(self.event['pid'], None)
            template = template.format(object_name, self.event['pid'], bbox)
        
        return template
    

class BboxEmbeddingTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        bbox = None
        object_name = "Person"

        if self.event['event_type'] == "COMPANION":
            template_merge = []
            pids = self.event['pids']
            for pid in pids:
                pid_bbox_embeddings = self.event.get('pid_bbox_embeddings', None)
                if pid_bbox_embeddings:
                    bbox = pid_bbox_embeddings.get(pid, None)
                    template_merge.append(template.format(object_name, pid, bbox))
            template = ",".join(template_merge)

        else:
            pid_bbox_embeddings = self.event.get('pid_bbox_embeddings', None)
            if pid_bbox_embeddings:
                bbox = pid_bbox_embeddings.get(self.event['pid'], None)
            template = template.format(object_name, self.event['pid'], bbox)
        
        return template
    

class RegionContextTemplate(Template):

    def __init__(self, template_file, event, area_descriptor=None):
        super().__init__(template_file, event, area_descriptor)

    @wrapper_str
    def __str__(self):
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        region = self.event['region']

        if self.event["event_type"] == "REGION_INOUT" and \
            self.event["region_type"] == "CAR":  # CAR_INOUT
            region_name = "Car"
        elif self.event["event_type"] == "REGION_VISIT":
            region_name = "Car"
        elif self.event['event_type'] == "STORE_INOUT":
            region_name = "Door"
        elif self.event["event_type"] == "REGION_INOUT" and \
            self.event["region_type"] == "INTERNAL_REGION":
            region_name = "InternalRegion"
        else:
            region_name = "Region"

        coords_round = []
        for coord in region.coords:
            coords_round.append([round(coord[0], 2), round(coord[1], 2)])
        template = template.format(
            region_name,
            region.id, 
            coords_round
        )
        
        return template
