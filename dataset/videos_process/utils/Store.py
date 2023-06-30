import json
import logging
import os.path as osp
from collections import defaultdict
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, Point
import numpy as np
import math
from utils.StoreRegion import FastVisibleBoundary, FastGridDistance

import pdb
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def check_car_pose_poly_order(pose_file):
    mlog.info("Checking the polygon order of car pose file at {}".format(pose_file))
    with open(pose_file, "rt") as f:
        pose = json.loads(f.read())
    for car in pose:
        coords = car["cords"]
        if len(coords) == 4:
            coords.append(coords[0])

        # assuming rectangle
        w = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
        h = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
        approx_area = w * h
        min_area_size = approx_area / 2.0

        poly = Polygon(coords)
        if poly.area <= min_area_size:
            return False
            # if not properly ordered (right-hand), area will be close to 0.
            mlog.warning("wrong ordered car pose identified {:.02f} vs {:.02f} <= {}".format(
                poly.area, approx_area,
                json.dumps(car)
            ))
            convex_cords = [[x, y] for x, y in zip(*poly.convex_hull.exterior.coords.xy)]
            car["cords"] = convex_cords[::-1][0:4]
        mlog.info("{}: area {:.02f} vs approx area {:.02f}".format(
            car["cid"], Polygon(car["cords"]).area, approx_area
        ))

    with open(pose_file, "wt") as f:
        f.write(json.dumps(pose, sort_keys=True, indent=2))
    return True

class FastGrid:
    def __init__(self, coords, cell_size):
        self._region = Polygon(coords)
        x = [pt[0] for pt in coords]
        y = [pt[1] for pt in coords]
        self._minX, self._maxX = min(x), max(x)
        self._minY, self._maxY = min(y), max(y)
        self._cell_size = int(cell_size)
        self._grid_x_size = int((self._maxX - self._minX) / self._cell_size)
        self._grid_y_size = int((self._maxY - self._minY) / self._cell_size)
        self._grid = np.zeros((self._grid_x_size + 1, self._grid_y_size + 1), dtype=np.bool)
        self._margins = set()
        # print(f"{self._minX}, {self._maxX}, {self._minY}, {self._maxY}, {self._grid_x_size}, {self._grid_y_size}")

    def create_grid(self):
        for i in range(0, self._grid_x_size + 1):
            x1 = i * self._cell_size + self._minX
            x2 = min((i + 1) * self._cell_size + self._minX, self._maxX)
            for j in range(0, self._grid_y_size + 1):
                y1 = j * self._cell_size + self._minY
                y2 = min((j + 1) * self._cell_size + self._minY, self._maxY)
                cur = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                self._grid[i, j] = False
                if self._region.contains(cur):
                    # print(f"{i}, {j} in the store region")
                    self._grid[i, j] = True
                elif self._region.intersects(cur):
                    self._margins.add((i, j))

    def contains(self, p):
        x, y = p
        if x < self._minX or x > self._maxX:
            return False
        if y < self._minY or y > self._maxY:
            return False
        i = int((x - self._minX) / self._cell_size)
        j = int((y - self._minY) / self._cell_size)
        if (i, j) in self._margins:
            return self._region.contains(Point(p))
        return self._grid[i, j]
    
    def contains_coords(self, p):
        x, y = p
        if x < self._minX or x > self._maxX:
            return False, (-1, -1)
        if y < self._minY or y > self._maxY:
            return False, (-1, -1)
        i = int((x - self._minX) / self._cell_size)
        j = int((y - self._minY) / self._cell_size)
        if (i, j) in self._margins:
            return self._region.contains(Point(p)), (i, j)
        return self._grid[i, j], (i, j)

    @property
    def region(self):
        return self._region

    @property
    def minX(self):
        return self._minX

    @property
    def minY(self):
        return self._minY

    @property
    def cell_size(self):
        return self._cell_size

    @property
    def grid(self):
        return self._grid


class Store(object):
    def __init__(self, storeInfo, meter2pixel=None, cameraInfo=None):
        ''' parse the given storeInfo folder,
            storeInfo has both parameters and area annotations
        '''
        self._store_info_path = storeInfo

        new_rule_fpath = osp.join(storeInfo, 'event_rules.json')
        area_fpath = osp.join(storeInfo, 'area_annotation.json')

        assert osp.exists(area_fpath), \
            "Must have area_annotation.json to get events!"
        assert osp.exists(new_rule_fpath), \
            "Must have event_rules.json to get events!"

        with open(area_fpath) as f:
            store_area = json.load(f)
        # Parse new rules format
        with open(new_rule_fpath) as f:
            rules = json.load(f)
        self._valid_events = list(rules.keys())
        self._staff_list = defaultdict(list)
        self._meter2pixel = None
        self._rules = rules
        # Select first available store region
        if "staff_type_map" in store_area:
            self._staff_type = store_area["staff_type_map"]
        else:
            self._staff_type = []
        # TODO: Modify to handle arbitrary number for INOUT

        # Parse region information
        self._has_car_region = False
        regions = {}
        self._regions = defaultdict(dict)
        self._internal_regions = []
        for region in store_area["region_areas"].values():
            region_type = store_area["region_type_map"][region["type"]]
            region_id = region["id"]
            region_poly = Polygon(region["coords"])
            if region_type == "STORE":
                if getattr(self, "_store_region", False):
                    raise Exception("Only support single store region!")
                region_poly = FastGrid(region["coords"], 100)
                region_poly.create_grid()
            elif region_type == "INTERNAL_ROOM":
                self._internal_regions.append({
                    "region": region_poly,
                    "name": region_id
                })
            else:
                regions[region["id"]] = region_poly
            self._regions[region_type][region_id] = region_poly
        num_store_regions = len(self._regions["STORE"])
        assert num_store_regions == 1, \
            f"Exactly 1 store region must be annotated! " \
            + f"(Found {num_store_regions})"
        self._counter_areas = regions

        # Search for car region in event_rules
        for event_type in rules:
            if "region_params" in rules[event_type]:
                if "CAR" in rules[event_type]["region_params"]:
                    self._has_car_region = True
                    break

        if self._has_car_region:
            mlog.info("This store has car regions! Estimate car sub regions for car visit/in-out")
            self._car_ids = []
            self._car_door_regions = {}
            self._infer_car_door_region(store_area)
            mlog.info("Done")
        else:
            self._car_ids = None
            self._car_door_regions = None
            mlog.info("This store has no car regions!")

        # Parse door information
        mlog.info("Reading door annotation information.")
        DEFAULT_WIDTH_RATIO = 0.2
        if "STORE_INOUT" not in rules:
            rules["STORE_INOUT"] = {}
        if "door_width_ratio" not in rules["STORE_INOUT"]:
            rules["STORE_INOUT"]["door_width_ratio"] = DEFAULT_WIDTH_RATIO
            mlog.info("Using default door_width_ratio = 0.2")
        self._doors = []
        for _, door in store_area["doors"].items():
            d_id = door["id"]
            door_type = store_area["door_type_map"][door["type"]]
            if door_type == "INTERNAL_DOOR":
                continue
            coords = door['coords']
            # Now completely switch to door line
            n = len(coords)
            length = 0
            for i in range(1, n):
                p1 = Point((coords[i - 1][0], coords[i - 1][1]))
                p2 = Point((coords[i][0], coords[i][1]))
                length += p1.distance(p2)
            length = length / (n - 1)
            if length > 1000:
                width = int(length * 0.05)
            else:
                width = int(length * rules["STORE_INOUT"]["door_width_ratio"])
            line = LineString(door['coords'])
            buffer = line.buffer(width, cap_style=2)
            self._doors.append(
                {
                    'region': buffer,
                    'name': d_id,
                    'type': door_type
                })
        mlog.info("Done.")
        if cameraInfo is not None:
            assert False, "Warning this option shows inferior result actually."
            # self._visible_area = FastVisibleBoundary(
            #     cameraInfo, cell_size=0.1, max_distance_from_camera=20)
        else:
            self._visible_area = None

        mlog.info("Reading meter2pixel information.")
        if meter2pixel is not None:
            self._meter2pixel = meter2pixel
            self._fast_distance2region = defaultdict(dict)
            for region in store_area["region_areas"].values():
                region_type = store_area["region_type_map"][region["type"]]
                region_id = region["id"]
                self._fast_distance2region.setdefault(region_type, {})
                self._fast_distance2region[region_type][region_id] = FastGridDistance(
                    region['coords'], cell_size=self._meter2pixel * 0.1, max_distance=self._meter2pixel * 5.0)

            for door in self.doors:
                name = door['name']
                self._fast_distance2region["DOOR"][name] = FastGridDistance(
                    door['region'], cell_size=self._meter2pixel * 0.1, max_distance=self._meter2pixel * 5.0)
        mlog.info("Done: {} pixels / meter".format(self._meter2pixel))
    def _infer_car_door_region(self, store_area, wheelbase_ratio=1.7):
        '''

        :param store_area:
        :param wheelbase_ratio: car_length / wheelbase
        :return:
        '''
        def _extend_line_by_dist(x, y, dist):

            len_xy = math.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
            m = x[0] + (y[0] - x[0]) / len_xy * dist
            n = x[1] + (y[1] - x[1]) / len_xy * dist
            return (m,n)

        carid_to_pose = {}
        for _, region in store_area['region_areas'].items():
            car_id = region['id']
            coords = region['coords']
            region_type = region['type']
            if region_type ==2 and not car_id.startswith('-'): # 2 --> car. https://code.aibee.cn/common/StoreInfos/-/blob/store_v7.1.0/VOLVO/jinan/xfh/area_annotation.json#L11
                assert len(coords) == 5, region
                mlog.info("car id: {} with coords: {}".format(car_id, coords))
                self._car_ids.append(car_id)
                carid_to_pose.setdefault(car_id, coords)

        for car_id in self._car_ids:

            car_poly = Polygon(carid_to_pose[car_id])
            upper_left, upper_right, lower_right, lower_left, _ = carid_to_pose[car_id]

            box = car_poly.minimum_rotated_rectangle
            x, y = box.exterior.coords.xy
            edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
            assert len(edge_length)==2, "edge lengths:{} for: {} coords: {}".format(edge_length, car_id, carid_to_pose[car_id])
            car_length = max(edge_length)

            front_length = car_length * 0.5 * (1 - 1 / wheelbase_ratio)
            back_length = front_length + 1/wheelbase_ratio*car_length

            door_upper_left = _extend_line_by_dist(upper_left, lower_left, front_length)
            door_lower_left = _extend_line_by_dist(upper_left, lower_left, back_length)

            door_upper_right = _extend_line_by_dist(upper_right, lower_right, front_length)
            door_lower_right = _extend_line_by_dist(upper_right, lower_right, back_length)

            door_polygon = Polygon([ door_upper_left, door_upper_right, door_lower_right, door_lower_left, door_upper_left])
            front_polygon = Polygon([ upper_left, door_upper_left, door_upper_right, upper_right])
            back_polygon = Polygon([ door_lower_left, door_lower_right, lower_right, lower_left])
            self._car_door_regions.setdefault(car_id,
                                              {
                                                  "front"   :   front_polygon,
                                                  "door"    :   door_polygon,
                                                  "back"    :   back_polygon
                                              }
                                              )

        # self._regions['CAR_DOOR'] = self._car_door_regions

    @property
    def car_ids(self):
        return self._car_ids

    @property
    def has_car_region(self):
        return self._has_car_region

    @property
    def fast_distance2region(self):
        return self._fast_distance2region

    @property
    def visible_area(self):
        return self._visible_area

    @property
    def regions(self):
        return self._regions

    @property
    def valid_events(self):
        return self._valid_events

    @property
    def rules(self):
        return self._rules

    @property
    def doors(self):
        return self._doors

    @property
    def staff(self):
        return self._staff_list

    @property
    def has_staff_types(self):
        if len(self._staff_list.keys()) == 1 and ("staff_class_1" in self._staff_list):
            # only default staff type without specification
            return False
        return True

    @staff.setter
    def staff(self, staff_list):
        if isinstance(staff_list, list):
            self._staff_list["staff_class_1"] = staff_list
        elif isinstance(staff_list, dict) and 'data' in staff_list:
            # IDS based staff
            actual_staff = staff_list["data"]
            for staff_details in actual_staff:
                try:
                    staff_pid = staff_details["user_id"]
                    staff_type = staff_details["additional"]["staff_type"]
                except TypeError:
                    raise TypeError(f"Invalid format for staff_list:"
                                    + f" {actual_staff}")
                if staff_type == "not_staff":
                    continue
                if (staff_type is not None
                        and not staff_type.startswith('staff_class_')):
                    raise ValueError(f"Invalid staff type {staff_type}")
                self._staff_list[staff_type].append(staff_pid)
        elif isinstance(staff_list, dict):
            # # TODO: need to sort this out with upstream modules
            # self._staff_list["staff_class_1"] = list(staff_list.keys())
            # return
            for k, v in staff_list.items():
                if "staff" in k:
                    staff_class = k
                    staffs = v
                    if staff_class == "not_staff":
                        continue
                    if (staff_class is not None
                            and not staff_class.startswith('staff_class_')):
                        raise ValueError(f"Invalid staff type {staff_class}")
                    self._staff_list[staff_class].extend(staffs)
                else:
                    staff = k
                    if v >= len(self._staff_type):
                        raise ValueError(f"Invalid staff type number {v}")
                    self._staff_list[self._staff_type[v]].extend([staff])

        else:
            raise NotImplementedError(
                f"Provided staff list ({type(staff_list)}) not supported!"
                )

    @property
    def meter2pixel(self):
        return self._meter2pixel

    @meter2pixel.setter
    def meter2pixel(self, val):
        assert val > 0, f"Meter2pixel value {val} must be > 0"
        self._meter2pixel = val

    @property
    def store_info_path(self):
        return self._store_info_path