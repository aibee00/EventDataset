import os

import vipy
from vipy.activity import Activity
from vipy.video import Scene



class LazyActivity(object):
    """ Load activity by activity meta infos.
    Load activity object when use it.
    Note: 
        不同于vipy.activity.Activity, 这里的Activity是指一个细粒度动作, 而不是一个视频.
        在对象上体现在它对应的是数据集路径下一个文件夹下的所有该动作的videos.
        该类主要是方便以某个'细粒度动作'为单位进行数据处理，例如，获取bbox,获取label等
        理论上，在没有load之前，该类只是提供一些方便的方法和预设的属性，不会有video实体，
        仅当我们需要处理某个activity时，通过load加载后，就可以方便地调用各个方法或属性直接获取信息
    """
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir
        self.V = None  # load activity with load bound method
        self._video_index = {}

    def _load_activity(self, activity_id):
        V = vipy.util.load(self.activity_path(activity_id))
        return V
    
    def activity_path(self, activity_id):
        """ Return a series of clips frames of a activity.
        :return:
        """
        activity_path = os.path.join(self.annotation_dir, activity_id + '.json')
        return activity_path
    
    @property
    def clip_files(self):
        """ Return clips list of this activity. """
        return [v.filename() for v in self.V]  # all clips dir of this activity
    
    @property
    def clip_names(self):
        """ Return a series of clips' names. """
        return [v.filename().split('/')[-1].split('.')[0] for v in self.V]  # all clips name of this activity 

    @property
    def framerates(self,):
        """ Return a series of clips' frames of a activity.
        :return:
        """
        f = [v.framerate() for v in self.V]  # video framerates 
        return f
    
    @property
    def bboxes(self,):
        """ Return a series of clips' bboxes of a activity.

        Return:
            List[List[vipy.object.BoundingBox]]
        
        Note:
            All bounding boxes in (xmin, ymin, width, height) format at video framerate (this will take a while)
            To get coords: bb.xywh()
            To get category: bb.category()
        """
        T = [[bb for t in v.tracklist() for bb in t] for v in self.V]
        return T
    
    @property
    def bbox_categories(self,):
        """ Return a series of clips' bboxes' category of a activity. 
        
        Return:
            List[List[Tuple[vipy.object.BoundingBox, str]]]
            List[List[Tuple[vipy.object.BoundingBox, str]]]
            List[List[Tuple[vipy.object.BoundingBox, str]]]
            ...
        :return:
        """
        T = [[(bb.category()) for t in v.tracklist() for bb in t] for v in self.V]
        return T
    
    @property
    def bbox_coords(self,):
        """ Return a series of clips' bboxes' coords of a activity.

        Return:
            List[List[Tuple[float, float, float, float]]], xywh
        """
        T = [[bb.xywh() for t in v.tracklist() for bb in t] for v in self.V]
        return T
    
    @property
    def annotations(self,):
        """ Return a series of clips' annotaions of a activity."""
        A = [[y for y in v.annotation()] for v in self.V]  # all framewise annotation (this will take a while)
        return A
    
    @property
    def index(self,):
        """ Return a video's index in this activity. """
        assert self._video_index, f"Please load activity first use self.load method."
        return self._video_index
    
    def get_framerate(self, video_name):
        """ Return a video's framerate. """
        return self.framerates[self.index[video_name]]
        
    def get_bboxes(self, video_name):
        """ Return a video's bboxes. """
        return self.bboxes[self.index[video_name]]
    
    def register(self,):
        """ 给每个video编号 """
        self._video_index = {v.filename().split('/')[-1].split('.')[0]: i for i, v in enumerate(self.V)}

    def load(self, activity_id):
        """
        Args:
            activity_id: str, activity id. e.g. person_enters_car
        Returns:
            None.
        Note:
            load activity object when use it.
            Note:
                不同于vipy.activity.Activity, 这里的Activity是指一个细粒度动作, 而不是一个视频.
                在对象上体现在它对应的是数据集路径下一个文件夹下的所有该动作的videos.
                该类主要是方便以某个'细粒度动作'为单位进行数据处理，例如，获取bbox,获取label等
        """
        # load data
        self.V = self._load_activity(activity_id)
        # register
        self.register()

