# PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset

## 简介
论文的官方资源PP HumanSeg:使用大规模电视会议视频数据集进行连接感知肖像分割。[[Paper](https://arxiv.org/abs/2112.07146) | [Poster](https://paddleseg.bj.bcebos.com/dygraph/humanseg/paper/12-HAD-poster.pdf) | [YouTube](https://www.youtube.com/watch?v=FlK8R5cdD7E)]

### Semantic Connectivity-aware Learning
SCL（Semantic Connectivity aware Learning，语义连接感知学习）框架，它引入了SC Loss（Semantin Connectivity-aware Loss）。SCL可以提高分割对象的完整性，提高分割精度。支持多类分割。[[Source code](../../paddleseg/models/losses/semantic_connectivity_loss.py)]

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148921096-29a4f90f-2113-4f97-87b5-19364e83b454.png" width="40%" height="40%">
</p>

### Connected Components Calculation and Matching

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148931627-bfaeeecb-c260-4d00-9393-a7e52a56ce18.png" width="40%" height="40%">
</p>

（a） 它表示预测和真值，即P和G。（b）连接分量分别通过CCL算法生成。（c） 使用IoU值匹配连接的组件。

### Segmentation Results

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148931612-bfc5a7f2-f6b7-4666-b2dd-86926ea7bfd7.png" width="60%" height="60%">
</p>

## 模型库

COCO数据集
|模型|骨干|输入尺寸|mIoU(Val)|Training Iters|模型下载|配置文件|
|-|-|-|-|-|-|-|
PP-HumanSeg | HRNet-W18 |  192x192  | 0.8276 | 20000 |  |  |

## 快速开始

### 数据集准备
* 下载数据集并且移动至`./dataset`
    ```
    dataset
    ├── train2017
    ├── val2017
    ├── test2017
    ├── annotations
    ├── COCO_person
    │   ├── train2017
    │   ├── val2017
    │   ├── annotations
    │   ├── label
    │   │   ├── train2017
    │   │   ├── val2017
    │   │   ├── train2017.txt
    │   │   └── val2017.txt

### 模型训练

PP-HumanSeg配置文件可见`configs/pp_humanseg/`.

```Shell
python train.py \
    --config configs/pp_humanseg/${model}.yml \
    --save_dir output/${model} \
    --do_eval \
    --use_vdl
```

## 模型评估

```shell
python val.py \
    --config configs/pp_humanseg/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams
```

## 模型预测
```shell
python predict.py \
    --config configs/pp_humanseg/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams \
    --image_path ${imagepath} or ${imagedir} \
    --save_dir output/result
```

## 引用

```latex
@InProceedings{Chu_2022_WACV,
    author    = {Chu, Lutao and Liu, Yi and Wu, Zewu and Tang, Shiyu and Chen, Guowei and Hao, Yuying and Peng, Juncai and Yu, Zhiliang and Chen, Zeyu and Lai, Baohua and Xiong, Haoyi},
    title     = {PP-HumanSeg: Connectivity-Aware Portrait Segmentation With a Large-Scale Teleconferencing Video Dataset},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {202-209}
}
```

## AI Studio项目传送门

https://aistudio.baidu.com/aistudio/projectdetail/4282050