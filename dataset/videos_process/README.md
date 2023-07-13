这个文件夹主要是对视频源文件的处理
例如，
- `download_videos.sh` : 
  - 从HDFS下载video, 但是这里不用先下载，因为在`extract_imgs_from_video_clean.py`中有"现用现下载"的功能。
- `create_data_dict_from_tpfp.py` : 
  - 根据tpfp下的`tps.json/fps_remaining.json`以及 `viz_infos/debug_viz_infos.json`, 生成 `data_dict.json`文件
- `create_data_infos_for_labeling.py` :
  - 创建视频中的pid相关的data_infos，
- `extract_imgs_from_video_clean.py` : 
  - 从video中按照一定的帧率提取images
  - 把pid的boundingbox贴附到各个channel对应的image上去
  - 保存最后的image
