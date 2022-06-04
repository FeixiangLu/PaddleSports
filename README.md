# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[when（什么时间）](#1-sportswhen)，[where（什么位置）](#2-sportswhere)，[who（是谁）](#3-sportswho)，[what（发生了什么）](#4-sportswhat)，[why（为什么）](#5-sportswhy)，[how（怎么样）](#6-sportshow)。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. AI模型：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [数据集](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[SportsBenchmark](#8-sportsbenchmark)，力争能够用一个数据集来评测所有算法模型。

4. [工具集](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [应用](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。


# 分模块介绍
该部分详细介绍“5W1H”各个模块的内容。

## 1. sports_when
&emsp; “when”模块重点从时域角度回答以下问题：

&emsp; 1）输入一段视频，首先判断是什么体育运动；

&emsp; 2）从一段视频中，精确分割出体育运动的起止时间；

&emsp; 3）判断每一帧属于哪个动作，以跳水三米板为例，动作过程分为：走板、起跳、空中、入水等阶段。

| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 1.when          | 1.1) 视频（时域）    | 视频分类（是什么体育项目）              | PP-TSM              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md         | 张孟希     |
|                 |                |                            | PP-TimeSformer      | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-timesformer.md | 张孟希     |
|                 |                |                            | SlowFast            | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md       | 张孟希     |
|                 |                |                            | AttentionLSTM       | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/attention_lstm.md | 张孟希     |
|                 |                |                            | MoViNet             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/movinet.md        | 张孟希     |
|                 |                | 片段切割（起始点，终止点）              | BMN                 | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md           | 张孟希     |
|                 |                | 动作识别（每一帧属于什么动作）            | MS-TCN              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/mstcn.md         | 张孟希     |
|                 |                |                            | ASRF                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/asrf.md          | 张孟希     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 2. sports_where

&emsp; “where”模块重点分析：前景（运动员）、背景（场馆）、相机，这三类对象的位置/位姿的信息：

&emsp; 1）运动员整体位姿：图像/视频中运动员的2D/3D定位，包含：2D/3D检测、2D分割、2D/3D跟踪等；

&emsp; 2）运动员局部位姿：运动员的骨骼姿态的分析，从粗粒度到细粒度，包含：2D骨骼关键点、2D骨骼姿态、3D骨骼姿态、2D-3D稠密映射、3D人体重建、3D人体动画等；

&emsp; 3）背景3D重建：利用多维传感器数据，1比1重建场馆的3D信息，相关技术包含：Simultaneous Localization and Mapping (SLAM)、Structure-from-Motion (SfM) 等；

&emsp; 4）相机6-DoF位姿：恢复相机的6-DoF位姿（位置xyz，旋转αβγ），有经典的PNP算法，以及深度模型算法。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 2.where         | 2.1) 2D检测      | 一阶段通用目标检测                  | PP-YOLOE            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe                            | 王成      |
|                 |                |                            | PP-PicoDet          | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet                            | 王成      |
|                 |                | 二阶段通用目标检测                  | Faster-RCNN         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        | 王成      |
|                 |                | 人体检测分析                     | PP-Human            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman                             | 王成      |
|                 |                |                            | PP-Pedestrian       | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/pedestrian                         | 王成      |
|                 |                | 水花/足球/篮球等小目标检测             | FPN                 | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        | 王成      |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.2) 2D分割      | 前景对象/背景分割                  | Mask-RCNN           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mask_rcnn                          | 张戈      |
|                 |                |                            | SOLOv2              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/solov2                             | 张戈      |
|                 |                |                            | PP-LiteSeg          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/pp_liteseg                               | 张戈      |
|                 |                |                            | DeepLabV3P          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p                               | 张戈      |
|                 |                | 交互式分割                      | EISeg               | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/EISeg                                            | 张戈      |
|                 |                | 人体分割                       | PP-HumanSeg         | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/contrib/PP-HumanSeg                              | 张戈      |
|                 |                | 人体毛发级精准分割                  | Matting             | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/Matting                                          | 张戈      |
|                 |                |                            | Human Matting       | https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/Matting/model/human_matting.py                   | 张戈      |
|                 |                | 视频目标分割                     | CFBI                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/cfbi.md          | 张戈      |
|                 |                |                            | MA-Net              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/EIVideo/EIVideo/docs/zh-CN/manet.md   | 张戈      |
|                 |                | 视频运动物体分割                   | Motion Segmentation |                                                                                                             | 张熙瑞     |
|                 |                | 视频人体分割 video matting       | BackgroundMattingV2 | https://github.com/PeterL1n/BackgroundMattingV2                                                             | 张戈      |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.3) 2D跟踪      | 人体跟踪                       | ByteTrack           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mot/bytetrack                      | 张熙瑞     |
|                 |                | 运动轨迹                       | PP-Tracking         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pptracking                          | 张熙瑞     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.4) 2D骨骼      | Top-Down                   | PP-TinyPose         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose                 | 张翰迪     |
|                 |                |                            | HR-Net              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint                           | 张翰迪     |
|                 |                | Bottom-Up                  | OpenPose            | https://github.com/CMU-Perceptual-Computing-Lab/openpose                                                    | 吕以豪     |
|                 |                |                            | MoveNet             | https://github.com/fire717/movenet.pytorch                                                                  | 吕以豪     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.5) 3D骨骼      | 单目                         | PP-TinyPose3D       |                                                                                                             | 张翰迪     |
|                 |                |                            | Position-based      |                                                                                                             | 吕以豪/张翰迪 |
|                 |                |                            | Angle-based         |                                                                                                             | 吕以豪/张翰迪 |
|                 |                |                            | 2D + Depth-based    |                                                                                                             | 吕以豪     |
|                 |                |                            | 2D + IK             |                                                                                                             | 吕以豪     |
|                 |                | 多目                         | Calibration         |                                                                                                             | 卢飞翔     |
|                 |                |                            | Fusion              |                                                                                                             | 卢飞翔     |
|                 |                | 深度相机                       | Kinect 3D Tracking  | https://docs.microsoft.com/zh-cn/azure/Kinect-dk/get-body-tracking-results                                  | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.6) 3D稠密关键点   | 2D-3D Dense Correspondence | DensePose           | https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose                                 | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.7) 3D人体重建    | Template Model             | SMPL                | https://smpl.is.tue.mpg.de/                                                                                 | 卢飞翔     |
|                 |                |                            | VIBE                | https://github.com/mkocabas/VIBE                                                                            | 卢飞翔     |
|                 |                |                            | PyMaf               | https://github.com/HongwenZhang/PyMAF                                                                       | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.8) SLAM      | 静态                         | 单目 ORB-SLAM...      | https://github.com/UZ-SLAMLab/ORB_SLAM3                                                                     | 卢飞翔     |
|                 |                |                            | 深度 KinectFusion...  | https://github.com/victorprad/InfiniTAM                                                                     | 卢飞翔     |
|                 |                |                            | 激光 LOAM             | https://github.com/RobustFieldAutonomyLab/LeGO-LOAM                                                         | 卢飞翔     |
|                 |                | 动态                         | DynamicFusion       | https://github.com/mihaibujanca/dynamicfusion                                                               | 卢飞翔     |
|                 |                |                            | DynSLAM             | https://github.com/AndreiBarsan/DynSLAM                                                                     | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 | 2.9) 相机6-DoF定位 | 内参                         | 张氏标定法               |                                                                                                             | 卢飞翔     |
|                 |                | 外参                         | 单张图像 PNP            |                                                                                                             | 卢飞翔     |
|                 |                |                            | 多张图像 SfM, SLAM      |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 3. sports_who


&emsp; “who”模块重点分析：图像/视频中有哪几类人员，分别是谁，特定人员在整场比赛的集锦等信息：

&emsp; 1）人员分类：把图像/视频中运动员、观众、裁判、后勤工作人员进行区分；

&emsp; 2）运动员识别：识别出特定运动员，包含：人脸识别、人体识别、号码簿识别等；

&emsp; 3）运动员比赛集锦：自动生成该运动员整场比赛的视频集锦。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 3.who           | 3.1) 识别        | 人脸检测                       | BlazeFace           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/face_detection                     | 王成      |
|                 |                | 人脸识别                       | Dlib                | http://dlib.net/                                                                                            | 王成      |
|                 |                | 基于人体的运动员识别                 |                     |                                                                                                             | 王成      |
|                 |                | 运动员Re-ID                   | MultiSports         | https://github.com/MCG-NJU/MultiSports                                                                      | 王成      |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 4. sports_what

&emsp; “what”模块重点分析体育比赛画面中呈现的信息，包含：运动、语音、视觉、多模态等：

&emsp; 1）运动属性，从视频前后帧信息推断运动信息，包含2D光流以及3D场景流相关技术；

&emsp; 2）语义属性，包含：图像/视频检索识别，视频动作识别，image/video caption等；

&emsp; 3）视觉属性，包含：画质增强，超分辨率，2D转3D，3D实时交互等；

&emsp; 4）多模态属性，视觉数据与语音数据、文本数据联合分析。


| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 4.what          | 4.1) 运动属性      | 2D optical flow            |                     |                                                                                                             | 张熙瑞     |
|                 |                | 3D scene flow              |                     |                                                                                                             | 张熙瑞     |
|                 | 4.2) 语义属性      | 图像检索识别                     | PP-Lite-Shitu       | https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/deploy/lite_shitu                               | 洪力      |
|                 |                |                            | PP-LCNetV2          | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 | 洪力      |
|                 |                | 视频动作识别                     | CTR-GCN             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/ctrgcn.md         | 洪力      |
|                 |                |                            | ST-GCN              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md          | 洪力      |
|                 |                |                            | AGCN                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/agcn.md           | 洪力      |
|                 |                | image caption              | COCO Caption        | https://github.com/tylin/coco-caption                                                                       | 王庆忠     |
|                 |                |                            | Im2Text             | https://www.cs.virginia.edu/~vicente/sbucaptions/                                                           | 王庆忠     |
|                 |                | video caption              | ActivityNet         | http://activity-net.org/challenges/2017/captioning.html                                                     | 王庆忠     |
|                 | 4.3) 视觉属性      | 3D Visualization           | Maya                |                                                                                                             | 卢飞翔     |
|                 |                |                            | Unity               |                                                                                                             | 卢飞翔     |
|                 |                |                            | Unreal              |                                                                                                             | 卢飞翔     |
|                 | 4.4) 多模态       | 文本+视觉                      | VideoBERT           |                                                                                                             | 王庆忠     |
|                 |                |                            | VisualBERT          |                                                                                                             | 王庆忠     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |

## 5. sports_why
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 5.why           | 5.1) 分析        | 技术、生理、心理、体能                |                     |                                                                                                             | 卢飞翔     |
|                 | 5.2) 推理        | 生物力学                       |                     |                                                                                                             | 卢飞翔     |
|                 | 5.3) 预测        | 内负荷、外负荷                    |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 6. sports_how
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 6.how           | 6.1) much      | 经费                         |                     |                                                                                                             | 卢飞翔     |
|                 | 6.2) many      | 人力                         |                     |                                                                                                             | 卢飞翔     |
|                 | 6.3) long      | 时间                         |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 7. data
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 7.data          | 7.1) 公开的数据集    |                            |                     |                                                                                                             | 王庆忠     |
|                 | 7.2) 自有的数据集    |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 7.3) 待构建的数据集   |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 8. sports_benchmark
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 9.benchmark     | 9.1) 训练数据集     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 9.2) 测试数据集     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 9.3) 评估脚本      |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |

## 9. tools
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 8.tools         | 8.1) 标注工具      |                            |                     |                                                                                                             | 张孟希     |
|                 | 8.2) 深度图生成工具   |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 10. applications
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 10.applications | 10.1) 足球       |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.2) 跳水       |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.3) 乒乓球      |                            |                     |                                                                                                             | 张孟希     |
|                 | 10.4) 花样滑冰     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.5) 健身       |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.6) 篮球       |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.7) 蹦床       |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.8) 大跳台      |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.9) 速度滑冰     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 10.10) 攀岩      |                            |                     |                                                                                                             | 卢飞翔     |


# 合作伙伴
- 国家队
- 央视
- 国家体育总局体育科学研究所
- 高校：北京大学，北京航空航天大学
- 体育类商业公司
- 世界冠军运动员、教练等

# 开发团队
- 百度研究院 机器人与自动驾驶实验室（RAL）

- 百度研究院 大数据实验室（BDL）

- 百度深度学习技术平台部（PaddlePaddle）





