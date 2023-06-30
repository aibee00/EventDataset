# 1. 仓库说明
该仓库用于创建EventGPT训练的数据集

# 2. 创建数据来源
数据来源目前来自于Store场景下
- Store Full Benchmark 的所有事件的GT
- Store Full Benchmark 的Video

目前使用GACNE_guangzhou_xhthwk_20210717的数据作为base测试  
以后可以扩展到其他Store的数据，甚至其他场景

# 3. How to run
>前提：假设我们已经从video中按照每秒钟一帧的方式转换成了各个channels的所有图片；  
>并按照channel和5分钟切片放到不同文件夹下，文件夹示例: [ch01010_20210728123500, ch01008_20210728094000, ....]  
>文件夹下的图片命名示例: [ch01001_20210728142051.jpg  ch01001_20210728142102.jpg  ch01001_20210728142109.jpg ...]  

***保存图片的文件夹结构 :***
```bash
videos
    |- GACNE-guangzhou-xhthwk-20210717
        |- ch01001_20210717101500
        |- ch01001_20210717102000
        |- ch01002_20210717132500
        ...
```

***运行 :***

```bash
bash run.sh
```


