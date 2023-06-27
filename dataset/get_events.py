"""
获取各个事件的信息
1. 进出店事件
2. 区域访问事件
3. 车区域访问事件
4. 进出车事件
5. 批次事件
6. 个人接待事件
"""

from abc import ABCMeta, abstractmethod
import sys
import os.path as osp
import json
import logging

sys.path.append(osp.join(osp.dirname(__file__), '..'))
sys.path.append(osp.join(osp.dirname(__file__), '../proto'))

from proto import store_events_pb2
from common import hms2sec, ts_to_string, get_location

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger("get_events Logger")
level = logging.getLevelName('INFO')
mlog.setLevel(level)

EVENTS_TYPE = {}
for evt_name, evt_index in store_events_pb2.EventType.items():
    print(evt_name, evt_index)
    EVENTS_TYPE.update({evt_index: evt_name})

REGIONS_TYPE = {}
for reg_name, reg_index in store_events_pb2.RegionType.items():
    print(reg_name, reg_index)
    REGIONS_TYPE.update({reg_index: reg_name})

# 这里对齐 自定义的名字 和 event_proto 中定义的名字
EVENT_MAP = {
        "region_visit" : store_events_pb2.EventType.REGION_VISIT,
        "region_inout" : store_events_pb2.EventType.REGION_INOUT,
        "car_inout" : store_events_pb2.EventType.REGION_INOUT,
        "store_inout" : store_events_pb2.EventType.STORE_INOUT,
        "group" : store_events_pb2.EventType.COMPANION,
        "individual_reception" : store_events_pb2.EventType.INDIVIDUAL_RECEPTION,
        "lookCar" : store_events_pb2.EventType.REGION_VISIT,
        "inCar" : store_events_pb2.EventType.REGION_INOUT,
    }

    
class EventInfoFactory(metaclass=ABCMeta):

    def __init__(self, events_file):
        self.events_file = events_file

    @abstractmethod
    def parse_event_info(self, events_file):
        """ 解析任务信息

        :return:
        """
        pass

    def get_event_info_by_type(self, event_type):
        """ 根据event_type获取目标任务的信息

        :param event_type: 任务类型 choose from:
            ["region_visit", "region_inout", "car_inout",
            "store_inout", "group", "individual_reception"]
        :return: 任务信息
        """
        # 将event_type转为小写
        event_type_lowcase = event_type.lower()

        event_type = EVENT_MAP.get(event_type_lowcase, None)

        assert event_type is not None, f"{event_type} not support!"

        if event_type in self.events_info:
            # 区别car_inout和region_inout
            if event_type_lowcase == "car_inout":
                car_inout_infos = {}
                for id, info in self.events_info[event_type].items():
                    if info["region_type"] == store_events_pb2.RegionType.CAR:
                        car_inout_infos.update({id: info})
                    else:
                        continue
                return car_inout_infos
            elif event_type_lowcase == "region_inout":
                region_inout_infos = {}
                for id, info in self.events_info[event_type].items():
                    if info["region_type"] != store_events_pb2.RegionType.CAR:
                        region_inout_infos.update({id: info})
                    else:
                        continue
                return region_inout_infos
            else:
                raise Exception(f"event_type:{event_type_lowcase} not found")
            
        else:
            raise Exception(f"event_type:{event_type}({event_type_lowcase}) not found")


class EventInfoFactoryImpl(EventInfoFactory):

    def __init__(self, events_file):
        super().__init__(events_file)
        self.events_info = self.parse_event_info(events_file)

    def parse_event_info(self, events_proto_file):
        """ 解析任务信息

        :return:{evttype: event_infos}
        """
        events_from_pb = store_events_pb2.Events()
        with open(events_proto_file, 'rb') as f:
            events_from_pb.ParseFromString(f.read())
        
        ret = {}
        for evt in events_from_pb.store_events:
            
            cid_info = evt.region.id.split(":")
            if len(cid_info) > 1:
                evt.region.id = cid_info[0]

            if evt.type == store_events_pb2.EventType.REGION_VISIT or \
                (evt.type == store_events_pb2.EventType.REGION_INOUT and evt.region.type == store_events_pb2.RegionType.CAR) or \
                (evt.type == store_events_pb2.EventType.REGION_INOUT and evt.region.type == store_events_pb2.RegionType.INTERNAL_REGION):
                
                ret.setdefault(evt.type, {})
                ret[evt.type].setdefault(evt.id, []).append(
                    {
                        "pid": evt.id,
                        "event_type": EVENTS_TYPE[evt.type],
                        "start_time": hms2sec(evt.start_time),
                        "end_time": hms2sec(evt.end_time),
                        "start_time_bj": ts_to_string(hms2sec(evt.start_time)),
                        "end_time_bj": ts_to_string(hms2sec(evt.end_time)),
                        "region_type": REGIONS_TYPE[evt.region.type],
                        "region_id": evt.region.id
                    }
                )
            
            elif evt.type == store_events_pb2.EventType.STORE_INOUT:
                ret.setdefault(evt.type, {})
                ret[evt.type].setdefault(evt.id, []).append(
                    {
                        "pid": evt.id,
                        "event_type": EVENTS_TYPE[evt.type],
                        "start_time": hms2sec(evt.start_time),
                        "end_time": hms2sec(evt.end_time),
                        "start_time_bj": ts_to_string(hms2sec(evt.start_time)),
                        "end_time_bj": ts_to_string(hms2sec(evt.end_time)),
                        "region_type": REGIONS_TYPE[evt.region.type],
                        "region_id": evt.region.id,
                        "in_door_id": evt.inout_properties.in_door_id,
                        "out_door_id": evt.inout_properties.out_door_id
                    }
                )

            elif evt.type == store_events_pb2.EventType.COMPANION:
                ret.setdefault(evt.type, {})
                ret[evt.type].setdefault(evt.id, []).append(
                    {
                        "group_id": evt.id,
                        "event_type": EVENTS_TYPE[evt.type],
                        "start_time": hms2sec(evt.start_time),
                        "end_time": hms2sec(evt.end_time),
                        "start_time_bj": ts_to_string(hms2sec(evt.start_time)),
                        "end_time_bj": ts_to_string(hms2sec(evt.end_time)),
                        "pids": list(evt.companion_properties.pid)
                    }
                )
            
            elif evt.type == store_events_pb2.EventType.INDIVIDUAL_RECEPTION:
                ret.setdefault(evt.type, {})
                ret[evt.type].setdefault(evt.id, []).append(
                    {
                        "pid": evt.id,
                        "staff_id": evt.reception_properties.staff_pid[0], 
                        "event_type": EVENTS_TYPE[evt.type],
                        "start_time": hms2sec(evt.start_time),
                        "end_time": hms2sec(evt.end_time),
                        "start_time_bj": ts_to_string(hms2sec(evt.start_time)),
                        "end_time_bj": ts_to_string(hms2sec(evt.end_time)),
                        "score": None,
                        "region_type": REGIONS_TYPE[evt.region.type],
                        "region_id": evt.region.id
                    }
                )

        return ret


def get_gt_region_visits(labels, region_visit_type):
    pid_inout = {}
    for idx, events in labels["shop"].items():
        pid = labels["int2str"][idx]
        pid_inout.setdefault(pid, [])
        for evt in events:
            if evt["type"] != region_visit_type:
                continue
            start_time = evt["video_start_time"]
            end_time = evt["video_end_time"]
            try:
                if region_visit_type == "region_visit":
                    pid_inout[pid].append(
                        {
                            "pid": pid,
                            "event_type": EVENTS_TYPE[EVENT_MAP[region_visit_type]],
                            "start_time": int(float(start_time)),
                            "end_time": int(float(end_time)),
                            "start_time_bj": ts_to_string(hms2sec(int(float(start_time)))),
                            "end_time_bj": ts_to_string(hms2sec(int(float(end_time)))),
                            "region_type": REGIONS_TYPE[store_events_pb2.RegionType.COUNTER],  # !注意这里指的是counter visit事件
                            "region_id": str(evt["region_id"])
                        }
                    )
                elif region_visit_type == "lookCar" or region_visit_type == "inCar":
                    # 看车事件，即car_visit; or 进出车事件，即car_inout
                    cid_info = str(evt["area"]).split(":")
                    if len(cid_info) > 1:
                        evt["area"] = cid_info[0]

                    pid_inout[pid].append(
                        {
                            "pid": pid,
                            "event_type": EVENTS_TYPE[EVENT_MAP[region_visit_type]],
                            "start_time": int(float(start_time)),
                            "end_time": int(float(end_time)),
                            "start_time_bj": ts_to_string(hms2sec(int(float(start_time)))),
                            "end_time_bj": ts_to_string(hms2sec(int(float(end_time)))),
                            "region_type": REGIONS_TYPE[store_events_pb2.RegionType.CAR],
                            "region_id": str(evt["area"])
                        }
                    )
                else:
                    mlog.warning("region visit event annotation error {}".format(evt))
                    continue
            except:
                mlog.warning("region visit event annotation error {}".format(evt))
                continue

    return pid_inout


def get_gt_pid_reception(labels):
    pid_reception = {}
    for idx, events in labels["reception"].items():
        pid = labels["int2str"][idx]
        pid_reception.setdefault(pid, [])
        for evt in events:
            if evt["type"] != "reception":
                mlog.info("Ignore event with type {}: {}".format(evt["type"], evt))
                continue
            staff_idx = evt["staff_pid"]
            if staff_idx in labels["int2str"]:
                staff_pid = labels["int2str"][staff_idx]
            elif str(staff_idx) in labels["int2str"]:
                staff_pid = labels["int2str"][str(staff_idx)]
            else:
                mlog.warning("Unknown staff idx {} identified for group reception {}: {} to {}".format(
                    staff_idx, pid,
                    ts_to_string(int(float(evt["video_start_time"]))),
                    ts_to_string(int(float(evt["video_end_time"]))),
                ))
                staff_pid = None
            start_time = evt["video_start_time"]
            end_time = evt["video_end_time"]
            try:
                pid_reception[pid].append(
                    {
                        "pid": pid,
                        "staff_id": staff_pid,
                        "event_type": EVENTS_TYPE[EVENT_MAP["individual_reception"]],
                        "start_time": int(float(start_time)),
                        "end_time": int(float(end_time)),
                        "start_time_bj": ts_to_string(hms2sec(int(float(start_time)))),
                        "end_time_bj": ts_to_string(hms2sec(int(float(end_time)))),
                        "score": None,
                        "region_type": REGIONS_TYPE[store_events_pb2.RegionType.STORE], 
                        "region_id": str(0)
                    }
                )
            except:
                mlog.warning("reception event annotation error {}".format(evt))
                continue

    return pid_reception

        
def get_gt_store_inouts(labels, popup_store=False):
    pid_inout = {}
    for idx, events in labels["shop"].items():
        pid = labels["int2str"][idx]
        pid_inout.setdefault(pid, [])
        for evt in events:
            if evt["type"] != "inOutStore":
                continue
            start_time = evt["video_start_time"]
            end_time = evt["video_end_time"]
            try:
                pid_inout[pid].append(
                    {
                        "pid": pid,
                        "event_type": EVENTS_TYPE[EVENT_MAP["store_inout"]],
                        "start_time": int(float(start_time)),
                        "end_time": int(float(end_time)),
                        "start_time_bj": ts_to_string(hms2sec(int(float(start_time)))),
                        "end_time_bj": ts_to_string(hms2sec(int(float(end_time)))),
                        "region_type": REGIONS_TYPE[store_events_pb2.RegionType.STORE],
                        "region_id": str(evt["id"]),
                        "in_door_id": "1" if popup_store else evt["door_in"],
                        "out_door_id": "1" if popup_store else evt["door_out"]
                    }
                )
            except:
                mlog.warning("inout event annotation error {}".format(evt))
                continue

    return pid_inout


class GTEventInfoFactoryImpl(EventInfoFactory):
    
    def __init__(self, gt_events_file, new_xy_path):
        super().__init__(gt_events_file)

        self.new_xy_path = new_xy_path
        self.events_info = self.parse_event_info(gt_events_file)

    def _update_events_info(self, events_info, event_type, event_infos):
        """
        :param events_info:
        :param event_type:
        :param event_infos: {pid: [events]}
        :return:
        """
        events_info.setdefault(event_type, {})

        for pid, event in event_infos.items():
            events_info[event_type].setdefault(pid, [])
            events_info[event_type][pid].extend(event)

    def get_gt_groups(self, labels):
        """ 这个之所以定义为类内函数是它需要用到轨迹信息来计算start/end time
        因为GT中没有标注start/end time

        :input: labels: json file
        :return: {group_id: [group_info]}
        """
        gt_groups = {}
        assert "groups" in labels, "groups not found in labels"
        for group in labels["groups"]:
            group_id = group["group_id"]
            
            gt_groups.setdefault(group_id, [])

            # 这里需要计算该group中所有pids的在店时间，该时间通过轨迹中的最小/最大时间获得
            # 初始化start_time/end_time
            group["start_time"] = float("inf")
            group["end_time"] = 0
        
            # 调用get_location函数获取轨迹信息
            for pid in group["pids"]:
                pid_file = osp.join(self.new_xy_path, pid + ".pb")
                pid_loc = get_location(pid_file)
                if pid_loc is None:
                    mlog.warning("pid {} not found in new_xy_path {}".format(pid, self.new_xy_path))
                    continue
                pid_loc = pid_loc[pid]
                ts_vec = sorted(pid_loc["loc"].keys())
                if len(ts_vec) == 0:
                    mlog.warning("pid {} has no location".format(pid))
                    continue
                group["start_time"] = min(group["start_time"], ts_vec[0])
                group["end_time"] = max(group["end_time"], ts_vec[-1])
            
            gt_groups[group_id].append(
                {
                    "group_id": group_id,
                    "event_type": EVENTS_TYPE[EVENT_MAP["group"]],
                    "start_time": int(group["start_time"]),
                    "end_time": int(group["end_time"]),
                    "start_time_bj": ts_to_string(hms2sec(int(group["start_time"]))),
                    "end_time_bj": ts_to_string(hms2sec(int(group["end_time"]))),
                    "pids": group["pids"],
                }
            )

        return gt_groups

    def parse_event_info(self, gt_events_file):
        """ 解析事件信息

        :input: gt_events_file: json file
        :return:{evttype: event_infos}

        特别说明:
            # region_visit, 这里指的是中量版的柜台访问事件(MI), 暂时不用
            # region_visits = get_gt_region_visits(labels, region_visit_type="region_visit")
            # region_inout, 这里我们跳过因为FullBmk中没有region inout事件
        """
        assert osp.exists(gt_events_file), f"{gt_events_file} not found!"

        with open(gt_events_file, 'r') as f:
            labels = json.load(f)

        events_info = {}

        # 解析store_inout
        store_inouts = get_gt_store_inouts(labels, popup_store=False)
        self._update_events_info(events_info, store_events_pb2.EventType.STORE_INOUT, store_inouts)

        # 解析car_visit
        car_visits = get_gt_region_visits(labels, region_visit_type="lookCar")
        self._update_events_info(events_info, store_events_pb2.EventType.REGION_INOUT, car_visits)

        # 解析car_inout
        car_inouts = get_gt_region_visits(labels, region_visit_type="inCar")
        self._update_events_info(events_info, store_events_pb2.EventType.REGION_INOUT, car_inouts)

        # 解析pid_reception
        individual_receptions = get_gt_pid_reception(labels)
        self._update_events_info(events_info, store_events_pb2.EventType.INDIVIDUAL_RECEPTION, individual_receptions)

        # 解析group
        gt_groups = self.get_gt_groups(labels)
        self._update_events_info(events_info, store_events_pb2.EventType.COMPANION, gt_groups)

        return events_info




