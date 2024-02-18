import os
import json
from pathlib import Path
from typing import Any



class ObjectsCounter(object):

    def __init__(self, label_path, save_path) -> None:
        """
        Args:
            label_path (str): Path to the label file. 
            save_path (str): Path to save the result.
            
            The label's format like this:
            {
                "image": "HONGQI-beijing-fkwd-20220109/ch01007_20220109134906.jpg",
                "caption": "['Person0<box>[947,547,1228,1252]</box>'] is the same group",
                "image_id": "ch01007_20220109134906",
                "task": "[TASK: 4]",
                "bbox": "person: [1831, 252, 2032, 808], person: [1591, 222, 1782, 751], car: [0, 227, 347, 529], tie: [1696, 295, 1731, 382], person: [762, 613, 1092, 1367], car: [611, 271, 1592, 975], person: [2259, 177, 2395, 583], person: [738, 209, 872, 550]",
                "caption_blip2": "a man is walking around a car showroom",
                "dense_caption": "black car with silver trim: [583, 256, 1578, 992]; a man walking in the street: [755, 613, 1083, 1356]; man wearing a suit: [1549, 210, 1771, 757]; a person is standing: [1807, 254, 2023, 811]; woman wearing orange jacket: [717, 208, 854, 548]; gray car is parked: [1, 219, 309, 525]; a large black sign: [1464, 0, 2166, 243]; a black and white informational sign: [7, 0, 1163, 88]; the jacket is orange in color: [717, 254, 841, 442]; black and white sign: [2251, 251, 2519, 787]; the windshield of a car: [855, 327, 1299, 625]; a group of people standing around: [0, 50, 2527, 1419]; black tire with silver rim: [1175, 737, 1285, 965]; a bunch of oranges: [1874, 996, 2198, 1365]; parking meter next to the car: [569, 429, 680, 709]; ",
                "prompt": "                      COMPANION events detecting: "
            },
            We need to do is count the objects in the key "bbox", like person, tie, car and so on.
            We need to return a set of all objects.

        """
        self.label_path = Path(label_path)
        self.label_dict = self.get_label_dict()
        self.object_counter = self.get_object_counter()

    def get_label_dict(self, ):
        assert self.label_path.exists(), f"Not Found! Please provide a valid path: {self.label_path.as_posix()}"
        return json.loads(Path(self.label_path).read_text())
    
    def get_object_counter(self):
        object_counter = {}
        for item in self.label_dict:
            bbox = item['bbox']
            for obj in bbox.split('], '):
                obj_name = obj.split(': ')[0]
                if obj_name == '':
                    continue
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
        # sort large to small
        object_counter = dict(sorted(object_counter.items(), key=lambda item: item[1], reverse=True))
        return object_counter
    
    def get_object_set(self):
        return list(self.object_counter.keys())
    
    def plot_distribution_of_objects(self,):
        import matplotlib.pyplot as plt
        # 定义画布为长条状
        plt.figure(figsize=(20, 10))
        # 设置标题
        plt.title('Distribution of Objects', fontdict={'size': 15, 'color': 'red'})
        # 设置 x 轴标签
        plt.xlabel('Objects', fontdict={'size': 15, 'color': 'red'})
        # 设置 y 轴标签
        plt.ylabel('Count', fontdict={'size': 15, 'color': 'red'})
        # 绘制柱状图，并设置颜色、透明度、边框和标签大小等参数
        plt.bar(self.object_counter.keys(), self.object_counter.values(), width=0.5, color='blue', alpha=0.8, align='center', edgecolor='black', linewidth=1.5)
        # 在每个柱子的顶端标注数目
        for x, y in zip(self.object_counter.keys(), self.object_counter.values()):
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
        
        # x 轴 label 倾斜 45 度
        plt.xticks(rotation=45)
        # save
        plt.savefig(f'{self.label_path.parent}/distribution_of_objects.png')
        plt.close()
        print(f'Save distribution of objects to {self.label_path.parent}/distribution_of_objects.png')
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Args:
            *args:
            **kwds:
        Returns:
            Set of all objects in the label file. 
            For example:
            {'person', 'car', 'tie', 'potted plant', 'car'}
        """
        # plot
        self.plot_distribution_of_objects()
        return self.get_object_set()
    


if __name__ == '__main__':
    label_path = '/training/wphu/Dataset/lavis/eventgpt/annotations/label_train.json'
    save_path = './'
    counter = ObjectsCounter(label_path, save_path)
    objects = counter()
    print(f"All {len(objects)} objects: {objects}")

