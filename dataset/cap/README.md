写在前面：这个文件夹主要是一些针对 Consented Activities of People (CAP) 数据集的一些处理脚本。
[github](https://github.com/visym/cap/tree/main)
---

# Consented Activities of People - Classification Training/Validation Set

[Consented Activities of People](https://visym.github.io/cap)


# Getting started

Follow the installation instructions for [vipy](https://github.com/visym/vipy). We recommend 

```python
pip install vipy[all]
```

To introspect videos:

```python
import vipy, vipy.dataset

V = vipy.util.load('/path/to/annotations.json')
h = vipy.dataset.Dataset(V).histogram()  # class frequencies
f = [v.framerate() for v in V]  # video framerates 
T = [[(bb, bb.category(), bb.xywh()) for t in v.tracklist() for bb in t] for v in V]  # all bounding boxes in (xmin, ymin, width, height) format at video framerate (this will take a while)
A = [[y for y in v.annotation()] for v in V]  # all framewise annotation (this will take a while)
```

To visualize a video:

```python
v = vipy.dataset.Dataset(V).take()  # take one at random
v.show()  # play it
v.annotate('annotated.mp4')  # save it
```


Refer to the [vipy documentation](https://visym.github.io/vipy) for advanced usage.


# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  


# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.


# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>


