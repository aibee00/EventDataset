import argparse
import os
import os.path as osp
import cv2
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

from descrption_generator import ImageObject, PromptDescriptor
from region_descriptor import AreaDescriptor
from gen_grid_cameras_map import FastGridRegion, get_best_camera_views

from get_events import EventInfoFactoryImpl, GTEventInfoFactoryImpl, EVENTS_TYPE
import json
import shutil

from common import STORE_INOUT_NEAR_TIME, duration, hms2sec, ts_to_string, get_location, get_overlap_time, get_pid_loc_time_range, logger


# 给出pid、location和时间段，返回覆盖到该location的所有channels
def get_cover_channels(grid_cameras_map, pid, loc, time_slice):
    """ 获取在时间段time_slice内pid所在位置有覆盖的cameras

    Args:
        grid_cameras_map: 预先生成的grid to cameras的映射关系
        pid: 某个人的pid
        loc: 轨迹文件
        time_slice: 可以是一段时间[st, et]或一个时刻ts
    Returns:
        channels: set of channels
    """
    channels = set()
    if isinstance(time_slice, list):
        for ts in range(time_slice[0], time_slice[1]):
            if pid not in loc:
                continue
            if ts not in loc[pid]['loc']:
                continue

            location = loc[pid]['loc'][ts]
            best_cameras = get_best_camera_views(location, grid_cameras_map)
            channels.update(set(best_cameras))
    else:
        if pid not in loc:
            return channels
        if time_slice not in loc[pid]['loc']:
            return channels

        location = loc[pid]['loc'][time_slice]
        best_cameras = get_best_camera_views(location, grid_cameras_map)
        channels.update(set(best_cameras))
    
    return channels


# 根据开始时间获取时间段
def get_video_time_range(start_time, offset):
    """
    start_time: 开始时间
    offset: 如[10, 20]
    :return: tuple(int, int)
    """
    if isinstance(start_time, str):
        start_time = hms2sec(int(start_time))
    end_time = start_time + offset
    return start_time, end_time


# 定义get_overlap_time_range函数，输入两个时间段，返回overlap时间段
def get_overlap_time_range(time_range1, time_range2):
    """
    time_range1: (start_time1, end_time1)
    time_range2: (start_time2, end_time2)
    """
    if time_range1[0] >= time_range2[1] or time_range1[1] <= time_range2[0]:
        return None
    start_time = max(time_range1[0], time_range2[0])
    end_time = min(time_range1[1], time_range2[1])
    return start_time, end_time
    

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

    def add_description(self, img_path, description):
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

    def update_dataset_path(self, new_dataset_path):
        """ 替换img_table中的img_object的path层空间
        """
        for img_name, img_object in self.img_table.items():
            img_object.path = osp.join(new_dataset_path, "imgs", img_name)


class EventDataset(Dataset):
    def __init__(self, store_info_path, car_pose, new_xy_path, events_file, video_path, dataset_path, grid_cameras_map_path):
        self.store_info_path = store_info_path
        self.car_pose = car_pose
        self.xy = new_xy_path

        self.events_file = events_file

        self.video_path = video_path
        self.dataset_path = dataset_path

        area_anno_path = osp.join(store_info_path, 'area_annotation.json')

        # 定义proto的查找表
        self.proto_table = {}

        # Get instance of event info factory
        # self.event_factory = EventInfoFactoryImpl(events_file)  # parse from events.pb
        self.event_factory = GTEventInfoFactoryImpl(events_file, self.xy)  # parse from gt

        # Get instacne of image descriptor
        self.img_descriptor = ImageDescriptor(self.dataset_path)

        # Get instacne of area descriptor
        self.area_descriptor = AreaDescriptor(area_anno_path, car_pose)

        # Get instacne of prompt descriptor
        self.prompt_descriptor = PromptDescriptor(osp.join(osp.dirname(__file__), "../prompt_templates"), self.area_descriptor)

        self.label_path = osp.join(self.dataset_path, "label.json")

        # Load grid_cameras_map
        with open(grid_cameras_map_path, 'r') as f:
            self.grid_cameras_map = json.load(f)

    @property
    def events_info(self):
        return self.event_factory.events_info
    
    def __len__(self):
        return len(self.img_descriptor.img_table)
    
    def get_all_time_covered_by_events(self):
        # 获取所有事件的开始时间和结束时间，并把时间段merge起来
        events = self.events_info
        time_covered = []
        for event_type, pid_events in events.items():
            logger.info(f"Merging events time of {EVENTS_TYPE[event_type]} ...\n")
            for pid, events in pid_events.items():
                for event in events:
                    start_time = event["start_time"]
                    end_time = event["end_time"]

                    # 如果是STORE_INOUT，则只需取进店后5s和出店的前5s的时间段
                    if EVENTS_TYPE[event_type] == "STORE_INOUT":
                        time_covered.append((start_time, start_time + 5))
                        time_covered.append((end_time - 5, end_time))
                        continue

                    # 同样地，如果是REGION_INOUT，则是需要取进区域和出区域的前后5s的时间段
                    if EVENTS_TYPE[event_type] == "REGION_INOUT":
                        time_covered.append((start_time - 5, start_time + 5))
                        time_covered.append((end_time - 5, end_time + 5))
                        continue

                    time_covered.append((start_time, end_time))
            
        # 把所有时间段merge起来
        time_covered = sorted(time_covered, key=lambda x:x[0])
        merged_time_covered = []
        for i in range(len(time_covered)):
            if i == 0:
                merged_time_covered.append(time_covered[i])
                continue
            start_time, end_time = time_covered[i]
            last_start_time, last_end_time = merged_time_covered[-1]
            if start_time <= last_end_time:
                merged_time_covered[-1] = (last_start_time, max(end_time, last_end_time))
                continue
            merged_time_covered.append(time_covered[i])

        return merged_time_covered

    # 根据时间，找出在这个时间发生的所有事件
    def get_events_in_time_slice(self, time_slice, margin=1):
        """
        :time_slice: (start_time, end_time)
        :return: events in time_slice
        :rtype: list[dict]
        """
        # 根据time_slice找出所有在这个时间段内发生的所有事件
        # events_in_time = []
        for event_type, pid_events in self.events_info.items():
            for pid, events in tqdm(pid_events.items(), desc=f"[{EVENTS_TYPE[event_type]}]"):
                for event in events:
                    start_time = event["start_time"]
                    end_time = event["end_time"]
                    # 判断time_slice与event时间段是否有overlap
                    if start_time >= time_slice[1] - margin or end_time <= time_slice[0] + margin:
                        continue

                    # logger.info(f"yiled {event_type} {pid} {event['start_time_bj']} {event['end_time_bj']}")
                    # events_in_time.append(event)
                    yield event
        # return events_in_time

    def save_img_descriptions(self,):
        """ 将生成的image的descriptions作为标注结果生成label.json
        """
        # 将生成的image的descriptions作为标注结果生成label.json
        with open(self.label_path, "w", encoding='utf-8') as f:
            json.dump(self.img_descriptor.img_table.value2str(), f, indent=2, ensure_ascii=False)

    def viz_img_prompt(self, num=10):
        """ 将label.json中的image和label可视化
        """
        # 定义draw_img_prompt函数，在同一画面中可视化img和annotation
        def draw_img_prompt(img, annotation):
            # 1. 将annotation解意为prompt
            prompt = self.prompt_descriptor.parse_annotation(annotation)
            # 2. 绘制prompt
            img = self.prompt_descriptor.draw_prompt(img, prompt)
            return img

        # 1. 读取label.json
        with open(self.label_path, "r", encoding='utf-8') as f:
            img_annos = json.load(f)
        
        # 2. 读取img和annotation
        for i, (img_name, annotation) in enumerate(tqdm(img_annos.items())):
            img_path = annotation["img"]
            img = cv2.imread(img_path)
            img = draw_img_prompt(img, annotation)
            img_dir = osp.join(self.dataset_path, "viz_img_prompt")
            if not osp.exists(img_dir):
                os.makedirs(img_dir)

            img_path = osp.join(img_dir, img_name)
            cv2.imwrite(img_path, img)
            print(f"Img saved to: {img_path}")
            
            if i == num:
                break
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    # 定义一个_get_location的函数，用以获取location
    def _get_location(self, pid, ts=None):
        """ 检查该proto是否已经在proto_table中，如果没有则调用get_location读取
        如果已经在查找表中，则直接返回
        """
        pid_proto_path = osp.join(self.xy, f"{pid}.pb")
        if pid not in self.proto_table.keys():
            self.proto_table[pid] = get_location(pid_proto_path)
        
        ret = None
        if ts is None:
            ret = self.proto_table[pid]
        else:
            ret = self.proto_table[pid][pid]['loc'][ts]
        
        return ret
    
    def get_valid_ts_set(self, event):
        """ 获取events的信息，获取事件的有效时间集合
        
        :return: (pid_list, valid_ts_set)
        """
        pid_list = []
        valid_ts_set = set()

        event_type = event["event_type"]

        if event_type == "COMPANION":
            # 如果是批次事件，我们只关注他们同时在店的情况
            pids = event["pids"]
            pid_list.extend(pids)

            valid_time_range = (event["start_time"], event["end_time"])
            for pid in pids:
                # 获取pid地处的location
                pic_locs = self._get_location(pid)
                # 通过pid轨迹的时间段
                time_slice = get_pid_loc_time_range(pic_locs, pid)

                # 如果time_slice长度小于5秒则跳过
                if duration(time_slice) < 5:
                    continue

                # common_time与time_slice的overlap时间段
                valid_time_range_n = get_overlap_time(valid_time_range, time_slice)

                # 如果valid_time_range和time_slice没有交集则两个时间段都保留
                if duration(valid_time_range_n) == 0:
                    valid_ts_set.update(set(range(*time_slice)))
                else:
                    valid_time_range = valid_time_range_n
            
            valid_ts_set.update(set(range(*valid_time_range)))

        elif event_type == "STORE_INOUT":
            # 如果是进出店，这里我们只需关注进店时的一小段时间即可，例如前后5s
            pid_list.append(event["pid"])
            valid_time_range_start = (event["start_time"] - STORE_INOUT_NEAR_TIME, event["start_time"] + STORE_INOUT_NEAR_TIME)
            valid_time_range_end = (event["end_time"] - STORE_INOUT_NEAR_TIME, event["end_time"] + STORE_INOUT_NEAR_TIME)
            valid_ts_set.update(set(range(*valid_time_range_start)))
            valid_ts_set.update(set(range(*valid_time_range_end)))

        else:
            pid_list.append(event["pid"])
            valid_time_range = (event["start_time"], event["end_time"])
            valid_ts_set.update(set(range(*valid_time_range)))

        return pid_list, valid_ts_set

    def _process_per_event(self, event):
        """ 处理每个事件
        处理过程:
            1.根据事件获取有效时刻和有效pid
            2.根据pid和时刻可以获得位置
            3.根据位置可以获得有效(有覆盖的)channels
            4.根据每个时刻ts的有效channels可以获得图片
            5.根据事件生成图片的描述
            6.将描述添加到所有channels的图片的描述中
        """
        # 1.根据事件获取有效时刻和有效pid
        valid_pids, valid_ts_set = self.get_valid_ts_set(event)

        # 处理valid_time_ranges中的每个ts
        for ts in valid_ts_set:
            for pid in valid_pids:
                # 2.根据pid和时刻可以获得位置
                locs = self._get_location(pid)

                # 3.根据位置可以获得有效(有覆盖的)channels
                channels = get_cover_channels(self.grid_cameras_map, pid, locs, ts)
                
                if channels is None:
                    continue

                # 4.根据每个时刻ts的有效channels可以获得图片
                imgs = []
                # bar = tqdm(channels, desc=f"Processing {len(channels)} channels")
                for channel in channels:
                    video_dirs = glob(osp.join(self.video_path, "{}_*".format(channel)))
                    for video_dir in video_dirs:
                        # logger.info(f"Processing video in: {video_dir}")
                        ch, time_str = video_dir.split("/")[-1].split("_")
                        date, time_start = time_str[:-6], time_str[-6:]
                        v_time_range = get_video_time_range(time_start, offset=5*60)

                        # 如果ts不在该video的时间段内，则跳过
                        if ts not in set(range(v_time_range[0], v_time_range[1])):
                            continue

                        ts_hms = ts_to_string(ts, sep="")

                        img_name = f"{ch}_{date}{ts_hms}.jpg"
                        img_file = osp.join(video_dir, img_name)
                        if not osp.exists(img_file):
                            continue
                        # img = cv2.imread(img_file)
                        imgs.append(img_file)

                # 将图片注册到descriptor中
                for img_file in imgs:
                    img_name = osp.basename(img_file)
                    self.img_descriptor.register(org_img_path=img_file, img_name=img_name)

                    # 5.根据事件生成图片的描述
                    description = self.prompt_descriptor.generate_prompt(event, ts)

                    # 6.将描述添加到所有channels的图片的描述中
                    self.img_descriptor.add_description(img_file, description)
    
    def create_dataset(self, ):
        """ 该函数用于生成dataset对应的imgs和labels

        如果数据集已经生成，则不用重复运行该函数；
        如果数据集尚未生成，运行该函数后，将会在self.dataset_path目录下生成：
            imgs: 用于保存所有含有事件的图片；
            label.json: 用于保存对所有图片的描述；
        
        运行该函数的前提：
            假设我们已经从video中按照每秒钟一帧的方式转换成了各个channels的所有图片；
            并按照channel和5分钟切片放到不同文件夹下，文件夹示例: [ch01010_20210728123500, ch01008_20210728094000, ....]
            文件夹下的图片命名示例: [ch01001_20210728142051.jpg  ch01001_20210728142102.jpg  ch01001_20210728142109.jpg ...]
        """

        # 获取有事件覆盖的所有时间段
        covered_time_clips = self.get_all_time_covered_by_events()

        # 迭代covered_time_clips中的每个时间段，获取每个time_clip时间段中发生的所有事件
        for time_clip in tqdm(covered_time_clips, desc="Processing time_clips"):
            # 获取事件，返回迭代器
            events_in_time_clip = self.get_events_in_time_slice(time_clip)

            # 迭代地获取每个events_in_time_clip的信息
            for event in tqdm(events_in_time_clip, desc=f"[{len(covered_time_clips)}clips]Processing events in each clip"):
                # 将事件转化为描述添加到各个channels的图片中去
                self._process_per_event(event)

        # 报存description标注结果
        self.save_img_descriptions()

    def __getitem__(self, index, random=False):
        if random:
            index = random.randint(0, len(self.img_descriptor.img_table)-1)
            img_name = self.img_descriptor.img_names[index]
        else:
            img_name = self.img_descriptor.img_names[index]
        
        img_obj = self.img_descriptor.img_table[img_name]
        img = cv2.imread(img_obj.path)
        description = img_obj.desc
        return img, description


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_info_path', type=str, help="camera_info_path", default="/ssd/wphu/CameraInfos/GACNE/guangzhou/xhthwk")
    parser.add_argument('--store_infos_path', type=str, help="store_infos_path", default="/ssd/wphu/StoreInfos/GACNE/guangzhou/xhthwk")
    parser.add_argument('--car_pose', type=str, help="path of pose.json", default="")
    parser.add_argument('--new_xy_path', type=str, help="path of new_xy", default="")
    parser.add_argument('--events_file', type=str, help="events.pb path or gt_file path", default="")
    parser.add_argument('--video_path', type=str, help="video path", default="")
    parser.add_argument('--dataset_path', type=str, help="output path", default="./data/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not osp.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    save_dir = osp.join(args.dataset_path, 'grid_cameras_map.json')

    # 首先创建grid_cameras_map文件，如果不存在则生成它
    if not osp.exists(save_dir):
        area_annotation_dir = osp.join(args.store_infos_path, 'area_annotation.json')
        area_annotation = json.loads(open(area_annotation_dir, 'rb').read())

        # Gen grid to cameras map
        coords = area_annotation["region_areas"]["STORE:0"]["coords"]
        store_region = FastGridRegion(coords)
        grid_cameras_map = store_region.gen_grid_to_cameras_map(args.camera_info_path)
        with open(save_dir, 'w') as f:
            json.dump(grid_cameras_map, f, indent=2)

    # 实例化Dataset类
    dataset_creater = EventDataset(
        args.store_infos_path,
        args.car_pose,
        args.new_xy_path, 
        args.events_file, 
        args.video_path, 
        args.dataset_path, 
        save_dir
    )

    # 如果数据集不存在，则创建它
    if True or not osp.exists(dataset_creater.label_path):
        dataset_creater.create_dataset()

    # 显示img和prompt
    # dataset_creater.viz_img_prompt(50)
    
    logger.info("Done!")



