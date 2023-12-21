""" 功能描述
用于数据扩增，For AibeeQA 数据集中的530张图
根据带标注的图片的name，去images中找到，这个时刻前后附近一个窗口范围内的相似的图片作为新样本，label复用已有的标注结果。
"""

import copy
import os
import os.path as osp
import json
import argparse
import bisect
import shutil

from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_result_v1_json', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v1_en.json', help='纯手工标注的gt')
    parser.add_argument('--train_img_list_v1', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/train_img_list.json')
    parser.add_argument('--label_result_v2_json', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/label_result_v2_en.json', help='借助gpt4v标注gt')
    parser.add_argument('--img_dir', type=str, default='/training/wphu/Dataset/lavis/eventgpt/images')
    parser.add_argument('--img_detections_json', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json')
    parser.add_argument('--save_dir', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--img_list_path', type=str, default='/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/train_img_list_v2.json')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--Filter_NUM', type=int, default=413)
    parser.add_argument('--EN_IOU_MATCH', action='store_true')
    parser.add_argument('--RESORT_ACCORDING_TO_IMG_LIST', action='store_true')
    args = parser.parse_args()
    return args


class DatasetExpander():
    def __init__(self, label_result_v1_json, train_img_list_v1, label_result_v2_json, img_dir, img_detections_json, save_dir, Filter_NUM, EN_IOU_MATCH = True, iou_threshold=0.5):
        self.Filter_NUM = Filter_NUM
        self.labels = self._load_labels(label_result_v1_json, train_img_list_v1, label_result_v2_json, Filter_NUM)
        self.img_dir = img_dir  # 这里由于 image_name中已经包含了images的路径，所以这里没有用上
        
        # Load images's detections
        self.detection_map = self.load_detections(img_detections_json)
        
        self.save_dir = Path(save_dir)

        self.EN_IOU_MATCH = EN_IOU_MATCH

        self.iou_threshold = iou_threshold

    def _load_labels(self, label_result_v1_json, train_img_list_v1, label_result_v2_json, Filter_NUM):
        labels_v1 = self.parse_label_result_v1(label_result_v1_json, train_img_list_v1)
        labels_v2 = json.loads(open(label_result_v2_json, 'r').read())

        labels = {}  # key is sample_id, value is label_infos

        # Move english label under the key 'caption' to key 'label'
        for i, (idx, label) in enumerate(labels_v2.items()):
            if int(idx) > Filter_NUM:
                continue

            label['label'] = label['caption']  # caption is en caption, label is ch caption
            labels[str(i)] = label

        # Merge both labels
        for i, (idx, label) in enumerate(labels_v1.items()):
            # 这里之所以使用 -idx 表示id，是为了不改变label_v2的id顺序，因为后面需要根据id过滤掉还没进行check的样本
            labels[str(-(idx + 1))] = label
        return labels

    @staticmethod
    def parse_label_result_v1(label_path, img_list_path):
        """ Parse label of v1 into v2 format
        """
        img_list = json.loads(open(img_list_path, 'r').read())
        img_list.sort()

        labels = json.loads(open(label_path, 'r').read())

        labels_tov2 = {}
        for i, (index, label) in enumerate(labels.items()):
            idx = int(index)
            img_dir = img_list[idx]

            labels_tov2[i] = {
                "caption": label,  # key is 'caption' means english label
                "label": label,  # key is 'label' means chinese label
                "global_caption": "",
                "img": img_dir
            }
        return labels_tov2

    @staticmethod
    def load_detections(img_detections_json):
        def parse_bbox_from_label(label):
            bboxes = []
            for bbox_str in label.split(';'):
                bbox_str = bbox_str.replace(' ', '')
                if bbox_str == '':
                    continue
                x1, y1, x2, y2 = bbox_str.split(':')[-1][1:-1].split(',')
                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            return bboxes
        
        img_detections = json.load(open(img_detections_json, 'r'))
        detection_map = {img_detections[i]['image']: parse_bbox_from_label(img_detections[i]['bbox']) for i in range(len(img_detections))}  # img_name to bboxes map
        return detection_map
    
    @staticmethod
    def get_full_path(root, cur_img_name):
        return os.path.join(root, f"{cur_img_name}.jpg")
    
    @staticmethod
    def _get_box_caption_map(label):
        """ parse bbox and corresponding caption.
        Args:
            label (str): the label of an image.
        Returns:
            box_caption_map (dict): the map of box and caption.
        """
        def parse_box_cap(caption):
            r_index = caption.find(']')
            if r_index == -1:
                return None, None
            
            l_index = caption.find('[')
            box = eval(caption[l_index:r_index + 1])
            cap = caption[r_index + 1:].strip()
            return box, cap
        
        box_caption_map = {}
        label = label.replace('\n\n', '\n')
        for caption in label.split('\n'):
            if caption == '':
                continue
            if caption.find(']') == -1:
                continue
            box, cap = parse_box_cap(caption)
            box_caption_map[tuple(box)] = cap
        return box_caption_map
    
    @staticmethod
    def compute_iou(box1, box2):
        """ Compute IoU of two boxes.
        Args:
            box1 (list): the first box.
            box2 (list): the second box.
        Returns:
            iou (float): the IoU of two boxes.
        """
        if not box1 or not box2:
            return 0
        
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        if x3 > x2 or x4 < x1 or y3 > y2 or y4 < y1:
            return 0
        else:
            inter_area = (min(x2, x4) - max(x1, x3)) * (min(y2, y4) - max(y1, y3))
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            return inter_area / (box1_area + box2_area - inter_area)
    
    def _get_new_img_path(self, image_path, new_image_dir=None):
        """ 从原始label中的img_path转换为新的img_name形式
        Input:
            image_path: 原文件路径 Path Like: "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500/1140.jpg"
            new_image_dir: 新文件目录，默认为None，则为保存到self.save_dir/images下。Path Like: '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images'

        Return:
            img_path: Path Like: '/training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/images/volvo_jinan_xfh_20210617_ch01019_20210617172500_1140.jpg'
        
        Note:
            old img_path: 指 label 中的image_path形式
        """
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        image_dir = image_path.parent
        img_name = image_path.name
        sub_paths = image_dir.as_posix().split('/')
        if new_image_dir is None:
            new_image_dir = self.save_dir / 'images'
            new_image_dir.mkdir(parents=True, exist_ok=True)
        new_img_path = new_image_dir / f'{sub_paths[-3]}__{sub_paths[-1]}__{img_name}'
        return new_img_path

    def filter_images_in_window(self, image_path, window=5):
        """ Filter images in window.
        Args:
            window (int): the window size for filtering. [ts - window, ts + window]
        Returns:
            None.
        Note:
            这里不同 site_id 的保存路径格式不一样
            v1: ['hongqi_beijing_fkwd_20220109', 'volvo_jinan_xfh_20210617', 'vw_changchun_rq_20210728', 'vw_hefei_zl_20210727', 'vw_tianjin_jz_20210727']
                images path: /training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500
                images names:
                0000.jpg  0084.jpg  1044.jpg  1128.jpg  1500.jpg  1572.jpg  1644.jpg  1872.jpg  1944.jpg  2016.jpg  2088.jpg  2160.jpg  2520.jpg  2592.jpg  2664.jpg  2940.jpg  3108.jpg
                0024.jpg  0972.jpg  1068.jpg  1140.jpg  1512.jpg  1584.jpg  1656.jpg  1884.jpg  1956.jpg  2028.jpg  2100.jpg  2352.jpg  2532.jpg  2604.jpg  2676.jpg  2952.jpg  3120.jpg
                0036.jpg  0984.jpg  1080.jpg  1440.jpg  1524.jpg  1596.jpg  1668.jpg  1896.jpg  1968.jpg  2040.jpg  2112.jpg  2364.jpg  2544.jpg  2616.jpg  2688.jpg  2964.jpg  3132.jpg
                0048.jpg  0996.jpg  1092.jpg  1464.jpg  1536.jpg  1608.jpg  1800.jpg  1908.jpg  1980.jpg  2052.jpg  2124.jpg  2376.jpg  2556.jpg  2628.jpg  2700.jpg  2976.jpg  3144.jpg
                0060.jpg  1008.jpg  1104.jpg  1476.jpg  1548.jpg  1620.jpg  1836.jpg  1920.jpg  1992.jpg  2064.jpg  2136.jpg  2388.jpg  2568.jpg  2640.jpg  2712.jpg  3084.jpg  3156.jpg
                0072.jpg  1020.jpg  1116.jpg  1488.jpg  1560.jpg  1632.jpg  1860.jpg  1932.jpg  2004.jpg  2076.jpg  2148.jpg  2508.jpg  2580.jpg  2652.jpg  2724.jpg  3096.jpg  3168.jpg

            v2: ['GACNE-guangzhou-xhthwk-20210717', 'HONGQI-beijing-fkwd-20220109', 'VW-changchun-rq-20210728']
                images path: /training/wphu/Dataset/lavis/eventgpt/images/HONGQI-beijing-fkwd-20220109/
                images names:
                ch01007_20220109114820.jpg  ch01007_20220109151015.jpg  ch01007_20220109172830.jpg  ch01008_20220109105402.jpg  ch01008_20220109163712.jpg  ch01010_20220109115012.jpg  ch01010_20220109184720.jpg
                ch01005_20220109140123.jpg  ch01005_20220109182806.jpg  ch01007_20220109114821.jpg  ch01007_20220109151016.jpg  ch01007_20220109173644.jpg  ch01008_20220109105403.jpg  ch01008_20220109163713.jpg  
                ch01010_20220109115013.jpg  ch01010_20220109184723.jpg ch01005_20220109140124.jpg  ch01005_20220109182807.jpg  ch01007_20220109114822.jpg ...
        """
        site_ids_v1 = ['hongqi_beijing_fkwd_20220109', 'volvo_jinan_xfh_20210617', 'vw_changchun_rq_20210728', 'vw_hefei_zl_20210727', 'vw_tianjin_jz_20210727']
        site_ids_v2 = ['GACNE-guangzhou-xhthwk-20210717', 'HONGQI-beijing-fkwd-20220109', 'VW-changchun-rq-20210728']

        image_name = image_path.stem  # "1140" or 'ch01007_20220109172830'
        if image_name.startswith('ch'):  # v2
            image_dir = image_path.parent  # "/training/wphu/Dataset/lavis/eventgpt/images/HONGQI-beijing-fkwd-20220109/"
            ch_name = image_name.split('_')[0]  # "ch01007"
            images = image_dir.glob(f'{ch_name}_*.jpg')
            images_names_map = {int(img.stem.split('_')[1]): img.stem for img in images}
            images_names = sorted(images_names_map.keys())
            cur_name = int(image_name.split('_')[1])
            index_start = bisect.bisect_left(images_names, cur_name - window)
            index_end = bisect.bisect_right(images_names, cur_name + window)
            cur_index = images_names.index(cur_name)
            images_names_expand = [images_names_map[name] for name in images_names[index_start:index_end]]
            left_expand = [images_names_map[name] for name in images_names[index_start:cur_index + 1]]  # include cur_index
            right_expand = [images_names_map[name] for name in images_names[cur_index:index_end]]  # include cur_index

        else:
            image_dir = image_path.parent  # "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500"
            images = image_dir.glob('*.jpg')
            images_names = sorted([int(img.stem) for img in images])
            cur_name = int(image_name)
            index_start = bisect.bisect_left(images_names, cur_name - window * 12)
            index_end = bisect.bisect_right(images_names, cur_name + window * 12)
            cur_index = images_names.index(cur_name)
            images_names_expand = [f"{img:04}" for img in images_names[index_start:index_end]]
            print(f"[ts_id-win, ts_id, ts_id+win]:  [{index_start} \t{cur_index} \t{index_end}]")
            left_expand = [f"{img:04}" for img in images_names[index_start:cur_index + 1]]  # include cur_index
            right_expand = [f"{img:04}" for img in images_names[cur_index:index_end]]  # include cur_index

        return images_names_expand, left_expand, right_expand
    
    
    def compute_cost_matrix(self, cur_bboxes, gt_bboxes):
        """ Compute the cost matrix of two sets of bboxes.
        Args:
            cur_bboxes (list): the bboxes of current image.
            gt_bboxes (list): the ground truth bboxes.
        Returns:
            cost_matrix (list): the cost matrix of two sets of bboxes.
        """
        cost_matrix = []
        for cur_bbox in cur_bboxes:
            cost = []
            for gt_bbox in gt_bboxes:
                cost.append(1 - self.compute_iou(cur_bbox, gt_bbox))
            cost_matrix.append(cost)
        return cost_matrix
    
    def iou_match(self, cur_bboxes, gt_bboxes):
        """ Match the bboxes with the ground truth bboxes.
        Args:
            cur_bboxes (list): the bboxes of current image.
            gt_bboxes (list): the ground truth bboxes.
        Returns:
            matched: True or nothing matched.
            row_ind: the matched row index.
            col_ind: the matched col index.
        """
        cost_matrix = self.compute_cost_matrix(cur_bboxes, gt_bboxes)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 需满足最小匹配阈值
        row_ind_filter, col_ind_filter = [], []
        num_not_matched = 0
        for rid, cid in zip(row_ind, col_ind):
            if 1- cost_matrix[rid][cid] > self.iou_threshold:
                row_ind_filter.append(rid)
                col_ind_filter.append(cid)
            else:
                num_not_matched += 1
                # print(f"Not matched!: [{1- cost_matrix[rid][cid]:.4f}]")
        print(f"num_not_matched in cur window: {num_not_matched} / {len(row_ind)}")
        row_ind, col_ind = row_ind_filter, col_ind_filter

        matched = True if len(row_ind) > 0 else False
        return matched, row_ind, col_ind

    def get_iou_matched_samples(self, image_path, left_expand, right_expand, label):
        """ Filter samples with matched labels.
        Args:
            left_expand (list): the left expanded images names.
            right_expand (list): the right expanded images names.
            label (dict): the label of current image.
        Returns:
            matched_imgs: the matched samples. Set((img, label), ...)
        """
        def get_bboxes_from_img_name(image_path, img_name):
            image_path = Path(image_path).parent / f"{img_name}.jpg"
            image_new_path = self._get_new_img_path(image_path).as_posix()
            if image_new_path not in self.detection_map:
                print(f"Skip, Not found: {image_new_path} in detections.")
                return None
            bboxes = self.detection_map[image_new_path]
            return bboxes
        
        def update_matched_imgs(matched_imgs, img_name, box_cap_map_ch, box_cap_map_en, pre_bboxes):
            # 获取当前要扩增的image对应的bboxes
            cur_bboxes = get_bboxes_from_img_name(image_path, img_name)

            if not cur_bboxes:
                return
            
            matched, row_indx, col_indx = self.iou_match(cur_bboxes, pre_bboxes)

            if matched:  # At least one was matched.
                matched_labels_ch = []
                matched_labels_en = []
                for row_ind, col_ind in zip(row_indx, col_indx):
                    if tuple(pre_bboxes[col_ind]) in  box_cap_map_ch:
                        matched_labels_ch.append(f"{cur_bboxes[row_ind]} {box_cap_map_ch[tuple(pre_bboxes[col_ind])]}")
                        box_cap_map_ch.update({tuple(cur_bboxes[row_ind]): box_cap_map_ch[tuple(pre_bboxes[col_ind])]})
                    
                    if tuple(pre_bboxes[col_ind]) in  box_cap_map_en:
                        matched_labels_en.append(f"{cur_bboxes[row_ind]} {box_cap_map_en[tuple(pre_bboxes[col_ind])]}")
                        box_cap_map_en.update({tuple(cur_bboxes[row_ind]): box_cap_map_en[tuple(pre_bboxes[col_ind])]})
                
                if matched_labels_ch and matched_labels_en:
                    new_label_ch = "\n".join(matched_labels_ch)
                    new_label_en = "\n".join(matched_labels_en)
                    matched_imgs.add((img_name, new_label_ch, new_label_en))

        # Begin
        matched_imgs = set()

        label_ch = label.get('label')
        label_en = label.get('caption')
        if label_ch is None:
            print(f"Skip, the key 'label' not found in label!")
            return matched_imgs
        
        if label_en is None:
            print(f"Skip, the key 'caption' not found in label!")
            return matched_imgs

        left_expand = list(reversed(left_expand))

        box_cap_map_ch = self._get_box_caption_map(label_ch)  # {(x1, y1, x2, y2): caption, ...}
        box_cap_map_en = self._get_box_caption_map(label_en)  # {(x1, y1, x2, y2): caption, ...}

        # 这里理论上 gt_bboxes_ch 和 gt_bboxes_en 一定是一样的，因为左右扩增的图片都是一样的
        gt_bboxes_ch = [list(box) for box in box_cap_map_ch.keys()]  # [[x1, y1, x2, y2], ...]
        gt_bboxes_en = [list(box) for box in box_cap_map_en.keys()]  # [[x1, y1, x2, y2], ...]

        for i, img_name in enumerate(left_expand):
            if i == 0:
                matched_imgs.add((img_name, label_ch, label_en))
                pre_bboxes = gt_bboxes_ch
                continue
            # pre_bboxes = get_bboxes_from_img_name(image_path, left_expand[i-1])
            update_matched_imgs(matched_imgs, img_name, box_cap_map_ch, box_cap_map_en, pre_bboxes)

        for i, img_name in enumerate(right_expand):
            if i == 0:
                pre_bboxes = gt_bboxes_ch
                continue
            # pre_bboxes = get_bboxes_from_img_name(image_path, right_expand[i-1])
            update_matched_imgs(matched_imgs, img_name, box_cap_map_ch, box_cap_map_en, pre_bboxes)
        
        return matched_imgs

    
    def expand_dataset(self, window=5, label_save_path=None):
        """ Expand dataset for fewshot learning.
        Args:
            window (int): the window size for expanding. [ts - window, ts + window]
        Returns:
            None.
        """
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        new_labels = {}
        sample_idx = 0

        for idx, label in tqdm(self.labels.items()):
            image_path = Path(label['img'])  # "/training/wphu/Dataset/lavis/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500/1140.jpg"
            
            images_names_expand, left_expand, right_expand = self.filter_images_in_window(image_path, window)

            if self.EN_IOU_MATCH:
                matched_expand_samples = self.get_iou_matched_samples(image_path, left_expand, right_expand, label)
                images_names_expand = matched_expand_samples

            for img_idx, item in enumerate(images_names_expand):
                if self.EN_IOU_MATCH:
                    aug_img_name, img_label_ch, img_label_en = item
                else:
                    aug_img_name = item
                
                aug_img_src_path = image_path.parent / f'{aug_img_name}.jpg'
                new_img_des_path = self._get_new_img_path(aug_img_src_path)  # 这里需要重新获取new_img_path

                # copy image
                if not new_img_des_path.exists():
                    print(f"Copying image {aug_img_src_path} to {new_img_des_path}")
                    shutil.copy(aug_img_src_path, new_img_des_path)
                else:
                    # print(f"Image {new_img_des_path} already exists, skipping...")
                    pass

                # update label to new_labels
                new_label = copy.deepcopy(label)
                new_label['img'] = new_img_des_path.as_posix()
                if self.EN_IOU_MATCH:
                    new_label['label'] = img_label_ch.replace('\n\n', '\n').replace('\n', '\n\n')  # chinese label
                    new_label['caption'] = img_label_en.replace('\n\n', '\n').replace('\n', '\n\n')  # englist label
                new_labels.update({sample_idx: new_label})
                sample_idx += 1
        
        if label_save_path is None:
            label_save_path = self.save_dir / 'train_label_result_aug.json'
        with open(label_save_path, 'w', encoding='utf-8') as f:
            json.dump(new_labels, f, indent=4, ensure_ascii=False)
        
        print(f"Total samples: {len(new_labels)}")
        print(f"Dataset expanded to {label_save_path}")

        return new_labels
    

    @staticmethod
    def resort_labels_according_to_img_list(img_list_path, new_labels, label_save_path):
        """ Resort labels according to img_list.
        这部分主要是为了在标注工具中查看标注结果，这需要根据 img_list 来重新调整 label 的index顺序，使得其与 img_list 顺序保持一致，因为我们模拟标注过程，标注顺序是按照img_list来的
        这里的 img_list 是用脚本`annotation_tools/gen_img_list.py` 生成的，它是一个list，每个元素是一个img_dir
        Args:
            new_labels (dict): new labels.
        Returns:
            None.
        """
        img_list = json.load(open(img_list_path))

        # get img_dir to label map
        img2label = {}
        for i, label in new_labels.items():
            img2label[label['img']] = label

        labels_final = {}
        num_skip = 0  # 记录跳过的图片数量
        for i, img_dir in enumerate(img_list):
            if img_dir not in img2label:
                num_skip += 1
                print(f"Image {img_dir} not found in label result, skipping...")  # This situation is no bbox in the img
                continue
            labels_final[i] = img2label[img_dir]

        # save labels_final
        with open(label_save_path, 'w', encoding='utf-8') as f:
            json.dump(labels_final, f, indent=4, ensure_ascii=False)
        
        print(f"Total skip images: {num_skip}")
        print(f"Total final samples: {len(labels_final)}")
        print(f"Dataset expanded to {label_save_path}")


if __name__ == '__main__':
    """ Note!!!
    Step1, You need to set EN_IOU_MATCH to False, then you can get all augmented images.
    Step2, You need to run detector with Aibee's detector to get all bboxes for all images. \
        This step need to run offline out of this script. Please refer to \
        http://wiki.aibee.cn/pages/viewpage.action?pageId=22731508 and https://aibee.feishu.cn/docs/doccnU3iGKbaQJFAK0cSKcciDNg .
        You can download the detector from `mms://singleview/bhrf_detection_store_retinanet_tf_gpu_general_20210419_v070000:v070000` .
    Step3, Run script `annotation_tools/parse_detections_from_weiding_result.py` to generate detections_result.json. 
        e.g. `python annotation_tools/parse_detections_from_weiding_result.py --input_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.txt --output_file /training/wphu/Dataset/lavis/eventgpt/fewshot_data_eventgpt/images_expand/detections/detection_result.json --use_provided_image_path`
    Step4, You need to run script `annotation_tools/gen_img_list.py` to generate img_list. 
    Step5, You need to set EN_IOU_MATCH to True and run this script, it will filter out matched bbox and generate corresponding labels.
    """
    # EN_IOU_MATCH = False
    # RESORT_ACCORDING_TO_IMG_LIST = False
    # Filter_NUM = 413  # 只标注到425, [0-413]训练集，[414~425]测试集

    args = args_parser()
    EN_IOU_MATCH = args.EN_IOU_MATCH
    RESORT_ACCORDING_TO_IMG_LIST = args.RESORT_ACCORDING_TO_IMG_LIST
    Filter_NUM = args.Filter_NUM  # 只标注到425, [0-413]训练集，[414~425]测试集

    label_save_path = osp.join(args.save_dir, 'train_label_result_v2_aug.json')  # save label result path.

    dataset_expander = DatasetExpander(
        args.label_result_v1_json,
        args.train_img_list_v1,
        args.label_result_v2_json,
        args.img_dir, 
        args.img_detections_json, 
        args.save_dir, 
        Filter_NUM=Filter_NUM,
        EN_IOU_MATCH=EN_IOU_MATCH,
        iou_threshold=args.iou_threshold
    )
    new_labels = dataset_expander.expand_dataset(window=args.window, label_save_path=label_save_path)


    if RESORT_ACCORDING_TO_IMG_LIST:
        new_labels = dataset_expander.resort_labels_according_to_img_list(args.img_list_path, new_labels, label_save_path)
    
    
