# 1. 仓库说明
该仓库用于创建EventGPT训练的数据集

# 2. 创建数据来源
数据来源目前来自于Store场景下
- Store Full Benchmark的所有事件的GT
- Store Full Benchmark 的Video

目前使用GACNE_guangzhou_xhthwk_20210717的数据作为base测试
以后可以扩展到其他Store的数据，甚至其他场景

# 3. How to run
>前提：假设我们已经把video处理成每秒1帧的图片

*图片的保存文件夹结构*
```bash
videos
    |- GACNE-guangzhou-xhthwk-20210717
        |- ch01001_20210717101500
        |- ch01001_20210717102000
        |- ch01002_20210717132500
        ...
```

```bash
bash run.sh
```


