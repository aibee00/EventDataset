import os
import json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from triangulation import Project3DTo2D

# area_annotation示例
"""
{
    "door_type_map": [
        "FRONT_DOOR",
        "BACK_DOOR",
        "INTERNAL_DOOR",
        "CONNECTOR"
    ],
    "region_type_map": [
        "STORE",
        "COUNTER",
        "CAR",
        "INTERNAL_ROOM",
        "INTERNAL_REGION",
        "TRANSACTION_REGION",
        "PRODUCT_REGION",
        "PUBLIC_SITTING_AREA",
        "STAFF_OFFICE",
        "RESTROOM",
        "EFFECTIVE_REGION",
        "COUNTER",
        "CASHIER",
        "ATM",
        "NON_FUNCTIONAL_COUNTER",
        "DOOR_REGION"
    ],
    "staff_type_map": [
        "not_staff",
        "staff_sales",
        "business_suit",
        "casual_suit",
        "cleaner",
        "guard",
        "mechanic",
        "poster"
    ],
    "doors": {
        "INTERNAL_DOOR:1": {
            "name": "",
            "id": "1",
            "type": 2,
            "coords": [
                [
                    2959,
                    115
                ],
                [
                    3150,
                    119
                ]
            ]
        },
        "FRONT_DOOR:2": {
            "name": "",
            "id": "2",
            "type": 0,
            "coords": [
                [
                    1044,
                    2070
                ],
                [
                    2650,
                    2070
                ]
            ]
        }
    },
    "region_areas": {
        "STORE:0": {
            "name": "",
            "id": "0",
            "type": 0,
            "coords": [
                [
                    605,
                    119
                ],
                [
                    2948,
                    119
                ],
                [
                    2945,
                    14
                ],
                [
                    3157,
                    18
                ],
                [
                    3168,
                    1674
                ],
                [
                    2869,
                    2066
                ],
                [
                    1037,
                    2070
                ],
                [
                    1044,
                    1505
                ],
                [
                    598,
                    1512
                ]
            ]
        },
        "INTERNAL_REGION:1": {
            "name": "",
            "id": "1",
            "type": 4,
            "coords": [
                [
                    2963,
                    112
                ],
                [
                    3154,
                    112
                ],
                [
                    3154,
                    22
                ],
                [
                    2959,
                    22
                ]
            ]
        },
        "INTERNAL_REGION:2": {
            "name": "",
            "id": "2",
            "type": 4,
            "coords": [
                [
                    1048,
                    1908
                ],
                [
                    1408,
                    1908
                ],
                [
                    1411,
                    1613
                ],
                [
                    1051,
                    1613
                ]
            ]
        }
    }
}
"""

# 车区域示例
"""
[
    {
        "cords": [
            [
                1906.0,
                1395.0
            ],
            [
                1543.0,
                1564.0
            ],
            [
                1072.0,
                608.0
            ],
            [
                1435.0,
                440.0
            ]
        ],
        "id": 1,
        "cid": 7,
        "image_url": [
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/1_9.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/1_19.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/1_12.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/1_29.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/1_2.jpg"
        ]
    },
    {
        "cords": [
            [
                2440.0,
                1843.0
            ],
            [
                2084.0,
                1631.0
            ],
            [
                2589.0,
                734.0
            ],
            [
                2945.0,
                946.0
            ]
        ],
        "id": 2,
        "cid": 6,
        "image_url": [
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/2_43.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/2_39.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/2_59.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/2_33.jpg",
            "https://car-model-identification.oss-cn-hangzhou.aliyuncs.com/GACNE_guangzhou_xhthwk/20210717/2_46.jpg"
        ]
    }
]
"""

class Area:
    def __init__(self, id, name, type, coords):
        self.id = id
        self.name = name
        self.type = type
        self.coords = coords

    @property
    def polygon(self):
        return self._get_polygon()
    
    def _get_polygon(self):
        return Polygon(self.coords)
    

class CarArea(Area):
    def __init__(self, id, name, type, coords, image_url, cid):
        super().__init__(id, name, type, coords)
        self.image_url = image_url
        self.cid = cid

    @property
    def polygon(self):
        return self._get_polygon()
    
    def _get_polygon(self):
        return Polygon(self.coords)


# 定义区域类型
class AreaDescriptor:

    def __init__(self, area_anno_path, car_pose, camera_info_path):
        self.area_anno_path = area_anno_path
        self.car_pose = car_pose
        self.camera_info_path = camera_info_path

        self.car_infos = self._load_car_region()

        self.area_annos = self._load_area_info()

        # Create Instance of Project3DTo2D
        store_tag = "/".join(Path(self.area_anno_path).parent.as_posix().split('/')[-3:])
        camera_info_path = Path(self.camera_info_path).parents[2]
        self.project = Project3DTo2D(camera_info_path, store_tag)

    # 加载车区域信息
    def _load_car_region(self):
        # 从self.car_pose json文件加載车区域信息
        car_region = {}
        with open(self.car_pose, 'r') as f:
            car_region = json.load(f)
        
        return car_region

    # 加载区域信息
    def _load_area_info(self):
        # 从self.area_anno_path json文件加载区域信息
        area_info = []
        with open(self.area_anno_path, 'r') as f:
            area_info = json.load(f)
        return area_info
    
    # 定义door_type_map属性
    @property
    def door_type_map(self):
        return self.area_annos['door_type_map']
    
    # 定义region_type_map属性
    @property
    def region_type_map(self):
        return self.area_annos['region_type_map']
    
    # 定义staff_type_map属性
    @property
    def staff_type_map(self):
        return self.area_annos['staff_type_map']
        
    # 定义INTERNAL_REGION属性
    @property
    def internal_region(self):
        internal_regions = {}
        for region in self.area_annos['region_areas']:
            if self.area_annos['region_areas'][region]['type'] == 4:
                internal_regions[self.area_annos['region_areas'][region]['id']] = \
                    Area(
                        self.area_annos['region_areas'][region]['id'], 
                        self.area_annos['region_areas'][region]['name'], 
                        self.area_annos['region_areas'][region]['type'], 
                        self.area_annos['region_areas'][region]['coords']
                    )
        
        return internal_regions
    
    # 定义STORE屬性
    @property
    def store_region(self):
        stores = {}
        for region in self.area_annos['region_areas']:
            if self.area_annos['region_areas'][region]['type'] == 0:
                stores[self.area_annos['region_areas'][region]['id']] = \
                    Area(
                        self.area_annos['region_areas'][region]['id'], 
                        self.area_annos['region_areas'][region]['name'], 
                        self.area_annos['region_areas'][region]['type'], 
                        self.area_annos['region_areas'][region]['coords']
                    )

        return stores
    
    # 定义doors属性
    @property
    def door_region(self):
        doors = {}
        for door in self.area_annos['doors']:
            doors[self.area_annos['doors'][door]['id']] = \
                Area(
                    self.area_annos['doors'][door]['id'], 
                    self.area_annos['doors'][door]['name'], 
                    self.area_annos['doors'][door]['type'], 
                    self.area_annos['doors'][door]['coords']
                )

        return doors

    # 定义Car属性
    @property
    def car_region(self):
        cars = {}
        # 从self.car_infos中获取信息，并初始化CarArea
        for car in self.car_infos:
            regname = car.get('name', f"CAR_{car['cid']}")
            regtype = car.get('type', 'CAR')

            car_area = CarArea(
                car['id'], 
                regname, 
                regtype, 
                car['cords'], 
                car['image_url'], 
                car['cid']
            )
            cars[car['id']] = car_area
            cars[str(car['cid'])] = car_area

        return cars
    
    def get_region_by_type_and_id(self, region_type, region_id):
        """ 标准化获取匿名对象
        """
        region_id = str(region_id)
        if 'DOOR' in region_type or 'STORE' in region_type:  # 注意这里Storeinout区域是door
            if region_id in self.door_region:
                return self.door_region[region_id]
            else:
                return None
        elif 'CAR' in region_type:
            return self.car_region[region_id]
        elif 'INTERNAL_REGION' in region_type:
            return self.internal_region[region_id]
        else:
            return None
        
    def map_floor_to_camera(self, points, channel):
        """ 将floor map上的点坐标映射到各个channel的camera坐标系

        :points: np.array, shape(n, 2)
        :return: np.array, shape(n, 2)
        """
        if isinstance(points, list):
            points = np.array(points)
        
        # Project floor coordinates to 3D
        points_3d = self.project.project_floor_to_3d(points)

        # Project 3D coordinates to Camera coordinates of channel
        points_cam = self.project.project_3d_to_2d(points_3d, channel)[:, 0, :]
        return points_cam.tolist()
