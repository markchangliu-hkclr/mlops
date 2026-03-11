# 目标

实现视觉检测/分割项目的数据和数据集的转化、变换、评估、可视化：
- 数据类型：bboxes，masks，polygons，contours，rles
- 标注格式：Labelme，YOLO，COCO
- 评估指标：mAP，precision，recall

## 上层决策

- 以实体作为模块，将该实体的类和同域函数（输入不牵扯其它类型，且没有副作用）放置于该模块下；类的 method 也必须为同域函数
- 通过函数实现逻辑，类尽量只用来组织数据（dataclass）或规范 API
- 代码库层深不超过2层

# 项目结构

```text
datasets
| - types.py
| - bboxes.py
| - masks.py
| - polygons.py
| - contours.py
| - rles.py
| - instances.py
| - labelme.py
| - coco.py
| - yolo.py
| - visualizers.py
| - evaluators.py
```
