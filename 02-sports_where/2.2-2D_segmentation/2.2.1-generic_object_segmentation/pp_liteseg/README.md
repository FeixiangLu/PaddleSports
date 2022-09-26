# PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model

## 简介

PP-LiteSeg，一种用于实时语义分割任务的新型轻量级模型。具体来说，PP-LiteSeg提出了一种灵活且轻量级的解码器 (FLD)，以减少先前解码器的计算开销。为了加强特征表示，使用了一个统一注意力融合模块（UAFM），它利用空间和通道注意力来产生权重，然后将输入特征与权重融合。此外，提出了一个简单的金字塔池模块（SPPM）以低计算成本聚合全局上下文。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148786-c8b91fd1-d006-4bad-8599-556daf959a75.png" width = "600" height = "300" alt="arch"  />
</div>

## 模型库

COCO数据集
|模型|骨干|输入尺寸|mIoU(Val)|模型下载|配置文件|
|-|-|-|-|-|-|
PP-LiteSeg | STDC1      |  520x520  | 0.8609 |  |  |


## 快速开始

### 数据集准备
* 下载数据集并且移动至`PaddleSeg/data`
    ```
    PaddleSeg/data
    ├── coco2017
    │   ├── train2017
    │   ├── val2017
    │   ├── annotations
    │   ├── train.txt
    │   └── val.txt
    ```

### 模型训练

PP-LiteSeg配置文件可见`PaddleSeg/configs/pp_liteseg/`.

```Shell
python train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --do_eval \
    --use_vdl
```

## 模型评估

```shell
python val.py \
    --config configs/pp_liteseg/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams
```

## 模型预测
```shell
python predict.py \
    --config configs/pp_liteseg/${model}.yml \
    --model_path output/${model}/best_model/model.pdparams \
    --image_path ${imagepath} or ${imagedir} \
    --save_dir output/result
```

## 引用
```
@article{2022PP,
  title={PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model},
  author={ Peng, J.  and  Liu, Y.  and  Tang, S.  and  Hao, Y.  and  Chu, L.  and  Chen, G.  and  Wu, Z.  and  Chen, Z.  and  Yu, Z.  and  Du, Y. },
  year={2022},
}
```