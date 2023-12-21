import cv2
from pathlib import Path

H, W = 1440, 2560  # 目前是这样的

def parse_bboxes_to_list_from_str(bboxes):
    bboxes_list = []
    if not bboxes:
        return bboxes_list
    
    for bbox in bboxes.split(';'):
        try:
            name, _box = bbox.split(':')
        except Exception as e:
            print(e)
            print(f"bbox: {bbox}")
            print(f"bboxes: {bboxes}")
            
        _box = _box.replace('[', '')
        _box = _box.replace(']', '')
        bbox_list = [float(x) for x in _box.split(',')]
        bboxes_list.append(bbox_list)
    return bboxes_list


def denorm(bboxes, H, W):
    """ denorm based on H,W

    H: image width
    W: image height
    """
    if not bboxes:
        return bboxes
    
    if not isinstance(bboxes[0], list):
        x1 = bboxes[0] * W
        y1 = bboxes[1] * H
        x2 = bboxes[2] * W
        y2 = bboxes[3] * H
        return [x1, y1, x2, y2]

    bboxes_list = []
    for bbox in bboxes:
        x1 = bbox[0] * W
        y1 = bbox[1] * H
        x2 = bbox[2] * W
        y2 = bbox[3] * H
        bboxes_list.append([x1, y1, x2, y2])
    return bboxes_list


def normalize(bboxes, H, W, dec=3):
    """ normalize based on H,W

    H: image width
    W: image height
    """
    bboxes_list = []
    for bbox in bboxes:
        x1 = round(bbox[0] / W, dec)
        y1 = round(bbox[1] / H, dec)
        x2 = round(bbox[2] / W, dec)
        y2 = round(bbox[3] / H, dec)
        bboxes_list.append([x1, y1, x2, y2])
    return bboxes_list


def plot_bboxes_on_image(image, bboxes, color=[0, 0, 255], linewidth=2):
    """
    Plot bboxes on image
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    for i, bbox in enumerate(bboxes):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, linewidth)
        cv2.putText(image, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return image


def get_label_info(image_path, label_map, info_name='bbox', use_relative_path=True):
    """ Get label info from image_path

    1. parse image_id from image_path
    2. get label_infos from label_map
    3. get bboxes from label_infos
    4. denorm bboxes
    5. return bboxes_list

    Args:
        image_path: image path
        label_map: label map
        info_name: info name, default bbox

    Returns:
        bboxes_list: list of bboxes
    """
    # Get image_id from image path, and get infos
    imgpath_obj = Path(image_path)

    if 'vw_' in imgpath_obj.as_posix() or 'volvo_' in imgpath_obj.as_posix() or 'hongqi_' in imgpath_obj.as_posix():
        relative_path = imgpath_obj.relative_to(imgpath_obj.parent.parent.parent.parent)
    else:        
        relative_path = imgpath_obj.relative_to(imgpath_obj.parent.parent)

    image_id = relative_path.as_posix() if use_relative_path else image_path  # Convert to posix path
    label_infos = label_map.get(image_id, {})
    bboxes = label_infos.get(info_name, "")
    bboxes_list = parse_bboxes_to_list_from_str(bboxes)
    return bboxes_list


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    return [x1, y1, w, h]
