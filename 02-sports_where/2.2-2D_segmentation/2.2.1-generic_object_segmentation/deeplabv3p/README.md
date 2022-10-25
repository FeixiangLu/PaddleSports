# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## 简介

DeepLabv3p，在DeepLabv3基础上进行了拓展。具体来说，通过添加一个简单但有效的解码器模块来细化分割结果，特别是沿着目标边界。DeepLabv3p进一步探索了Xception模型，并将深度可分离卷积应用于Atrous空间金字塔池和解码器模块，从而获得更快、更强的编码器-解码器网络。

## 模型库

COCO数据集
|模型|骨干|输入尺寸|mIoU(Val)|Training Iters|模型下载|配置文件|
|-|-|-|-|-|-|-|
DeepLabv3p | ResNet50_OS8   |  520x520  | 0.8686 | 40000 |  |  |
DeepLabv3p | ResNet101_OS8  |  520x520  | 0.7191 | 40000 |  |  |


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
    ```

### 模型训练

DeepLabv3p配置文件可见`configs/deeplabv3p/`.

```Shell
python train.py \
    --config configs/deeplabv3p/${model}.yml \
    --save_dir output/${model} \
    --do_eval \
    --use_vdl
```

## 模型评估

```shell
python val.py \
    --config configs/deeplabv3p/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams
```

## 模型预测
```shell
python predict.py \
    --config configs/deeplabv3p/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams \
    --image_path ${imagepath} or ${imagedir} \
    --save_dir output/result
```

## 引用
```
@article{2018Encoder,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={ Chen, L. C.  and  Zhu, Y.  and  Papandreou, G.  and  Schroff, F.  and  Adam, H. },
  journal={Springer, Cham},
  year={2018},
}
```

## AI Studio项目传送门

https://aistudio.baidu.com/aistudio/projectdetail/4664992