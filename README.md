# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[*when*](#1-when)（什么时间），[*where*](#2-where)（什么位置），[*who*](#3-who)（是谁），[*what*](#4-what)（发生了什么），[*why*](#5-why)（为什么），[*how*](#6-how)（怎么样）。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. *AI模型*：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [*数据*](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[*SportsBenchmark*](#8-benchmark)，力争能够用一个数据集来评测所有算法模型。

4. [*工具*](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [*应用*](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。


# 分模块介绍
该部分详细介绍“5W1H”各个模块的内容。

## 1. when
&emsp; “when”模块重点从时域角度回答以下问题：

&emsp; 1）输入一段视频，首先判断是什么体育运动；

&emsp; 2）从一段视频中，精确分割出体育运动的起止时间；

&emsp; 3）判断每一帧属于哪个动作，以跳水三米板为例，动作过程分为：走板、起跳、空中、入水等阶段。

&emsp; 4）时间同步，针对多相机同步问题，整理了硬件同步和软件同步两种控制方法。

&emsp; 5）编解码，包括视频编解码和音频编解码。

| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 1.when          | 1.1) 视频分类          | 视频分类（是什么体育项目）               | PP-TSM                                              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md         | 张孟希     |
|                 |                    |                             | PP-TimeSformer                                      | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-timesformer.md | 张孟希     |
|                 |                    |                             | SlowFast                                            | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md       | 张孟希     |
|                 |                    |                             | AttentionLSTM                                       | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/attention_lstm.md | 张孟希     |
|                 |                    |                             | MoViNet                                             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/movinet.md        | 张孟希     |
|                 | 1.2) 视频分割          | 片段切割（起始点，终止点）               | BMN                                                 | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md           | 张孟希     |
|                 | 1.3) 视频理解          | 动作识别（每一帧属于什么动作）             | MS-TCN                                              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/mstcn.md         | 张孟希     |
|                 |                    |                             | ASRF                                                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/asrf.md          | 张孟希     |
|                 | 1.4) 硬件同步          | 硬件同步                        | PTP同步，IEEE 1588                                     | 选择支持GigE Vision 2.0协议的相机                                                                                    | 卢飞翔     |
|                 |                    | 软件同步                        | CPU时钟同步                                             |                                                                                                             | 卢飞翔     |
|                 | 1.5) 编解码           | 视频编码                        | H.264/MPEG-4 AVC                                    |                                                                                                             | 卢飞翔     |
|                 |                    | 音频编码                        | WAV/MP3/AAC                                         |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 2. where

&emsp; “where”模块重点分析：前景（运动员）、背景（场馆）、相机，这三类对象的位置/位姿的信息：

&emsp; 1）运动员整体位姿：图像/视频中运动员的2D/3D定位，包含：2D/3D检测、2D分割、2D/3D跟踪等；

&emsp; 2）运动员局部位姿：运动员的骨骼姿态的分析，从粗粒度到细粒度，包含：2D骨骼关键点、2D骨骼姿态、3D骨骼姿态、2D-3D稠密映射、3D人体重建、3D人体动画等；

&emsp; 3）背景3D重建：利用多维传感器数据，1比1重建场馆的3D信息，相关技术包含：Simultaneous Localization and Mapping (SLAM)、Structure-from-Motion (SfM) 等；

&emsp; 4）相机6-DoF位姿：恢复相机的6-DoF位姿（位置xyz，旋转αβγ），有经典的PNP算法，以及深度模型算法。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 2.where         | 2.1) 2D检测          | 一阶段通用目标检测                   | PP-YOLOE                                            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe                            | 王成      |
|                 |                    |                             | PP-PicoDet                                          | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet                            | 王成      |
|                 |                    | 二阶段通用目标检测                   | Faster-RCNN                                         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        | 王成      |
|                 |                    | 人体检测分析                      | PP-Human                                            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman                             | 王成      |
|                 |                    |                             | PP-Pedestrian                                       | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/pedestrian                         | 王成      |
|                 |                    | 水花/足球/篮球等小目标检测              | FPN                                                 | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        | 王成      |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.2) 2D分割          | 前景对象/背景分割                   | Mask-RCNN                                           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mask_rcnn                          | 张戈      |
|                 |                    |                             | SOLOv2                                              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/solov2                             | 张戈      |
|                 |                    |                             | PP-LiteSeg                                          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/pp_liteseg                               | 张戈      |
|                 |                    |                             | DeepLabV3P                                          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p                               | 张戈      |
|                 |                    | 交互式分割                       | EISeg                                               | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/EISeg                                            | 张戈      |
|                 |                    | 人体分割                        | PP-HumanSeg                                         | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/contrib/PP-HumanSeg                              | 张戈      |
|                 |                    | 人体毛发级精准分割                   | Matting                                             | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/Matting                                          | 张戈      |
|                 |                    |                             | Human Matting                                       | https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/Matting/model/human_matting.py                   | 张戈      |
|                 |                    | 视频目标分割                      | CFBI                                                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/cfbi.md          | 张戈      |
|                 |                    |                             | MA-Net                                              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/EIVideo/EIVideo/docs/zh-CN/manet.md   | 张戈      |
|                 |                    | 视频运动物体分割                    | Motion Segmentation                                 |                                                                                                             | 张熙瑞     |
|                 |                    | 视频人体分割 Video Matting        | BackgroundMattingV2                                 | https://github.com/PeterL1n/BackgroundMattingV2                                                             | 张戈      |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.3) 2D跟踪          | 人体跟踪                        | ByteTrack                                           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mot/bytetrack                      | 张熙瑞     |
|                 |                    | 运动轨迹                        | PP-Tracking                                         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pptracking                          | 张熙瑞     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.4) 2D骨骼          | Top-Down                    | PP-TinyPose                                         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose                 | 张翰迪     |
|                 |                    |                             | HR-Net                                              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint                           | 张翰迪     |
|                 |                    | Bottom-Up                   | OpenPose                                            | https://github.com/CMU-Perceptual-Computing-Lab/openpose                                                    | 吕以豪     |
|                 |                    |                             | MoveNet                                             | https://github.com/fire717/movenet.pytorch                                                                  | 吕以豪     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.5) 3D骨骼          | 单目                          | PP-TinyPose3D                                       |                                                                                                             | 张翰迪     |
|                 |                    |                             | Position-based                                      |                                                                                                             | 吕以豪/张翰迪 |
|                 |                    |                             | Angle-based                                         |                                                                                                             | 吕以豪/张翰迪 |
|                 |                    |                             | 2D + Depth-based                                    |                                                                                                             | 吕以豪     |
|                 |                    |                             | 2D + IK                                             |                                                                                                             | 吕以豪     |
|                 |                    | 多目                          | Calibration                                         |                                                                                                             | 卢飞翔     |
|                 |                    |                             | Fusion                                              |                                                                                                             | 卢飞翔     |
|                 |                    | 深度相机                        | Kinect 3D Tracking                                  | https://docs.microsoft.com/zh-cn/azure/Kinect-dk/get-body-tracking-results                                  | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.6) 2D/3D稠密映射     | 2D-2D Dense Correspondences | DeepMatching                                        | http://lear.inrialpes.fr/src/deepmatching/                                                                  | 卢飞翔     |
|                 |                    | 2D-3D Dense Correspondences | DensePose                                           | https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose                                 | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.7) 3D人体重建        | Template Model              | SMPL                                                | https://smpl.is.tue.mpg.de/                                                                                 | 卢飞翔     |
|                 |                    |                             | VIBE                                                | https://github.com/mkocabas/VIBE                                                                            | 卢飞翔     |
|                 |                    |                             | PyMaf                                               | https://github.com/HongwenZhang/PyMAF                                                                       | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.8) SLAM          | 静态                          | 单目 ORB-SLAM...                                      | https://github.com/UZ-SLAMLab/ORB_SLAM3                                                                     | 卢飞翔     |
|                 |                    |                             | 深度 KinectFusion...                                  | https://github.com/victorprad/InfiniTAM                                                                     | 卢飞翔     |
|                 |                    |                             | 激光 LOAM                                             | https://github.com/RobustFieldAutonomyLab/LeGO-LOAM                                                         | 卢飞翔     |
|                 |                    | 动态                          | DynamicFusion                                       | https://github.com/mihaibujanca/dynamicfusion                                                               | 卢飞翔     |
|                 |                    |                             | DynSLAM                                             | https://github.com/AndreiBarsan/DynSLAM                                                                     | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 | 2.9) 相机6-DoF定位     | 内参                          | 张氏标定法                                               |                                                                                                             | 卢飞翔     |
|                 |                    | 外参                          | 单张图像 PNP                                            |                                                                                                             | 卢飞翔     |
|                 |                    |                             | 多张图像 SfM, SLAM                                      |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 3. who


&emsp; “who”模块重点分析：图像/视频中有哪几类人员，分别是谁，特定人员在整场比赛的集锦等信息：

&emsp; 1）人员分类：把图像/视频中运动员、观众、裁判、后勤工作人员进行区分；

&emsp; 2）运动员识别：识别出特定运动员，包含：人脸识别、人体识别、号码簿识别等；

&emsp; 3）运动员比赛集锦：自动生成该运动员整场比赛的视频集锦。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 3.who           | 3.1) 人员分类          | 运动员、裁判、观众、后勤人员              | PP-LCNetV2.md                                       | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 | 王成      |
|                 | 3.2) 运动员识别         | 人脸检测                        | BlazeFace                                           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/face_detection                     | 王成      |
|                 |                    | 人脸识别                        | Dlib                                                | http://dlib.net/                                                                                            | 王成      |
|                 |                    | 基于人体的运动员识别                  |                                                     |                                                                                                             | 王成      |
|                 | 3.3) “一人一档”        | 运动员Re-ID                    | MultiSports                                         | https://github.com/MCG-NJU/MultiSports                                                                      | 王成      |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 4. what

&emsp; “what”模块重点分析体育比赛画面中呈现的信息，包含：运动、语音、视觉、多模态等：

&emsp; 1）运动属性，从视频前后帧信息推断运动信息，包含2D光流以及3D场景流相关技术；

&emsp; 2）语义属性，包含：图像/视频检索识别，视频动作识别，image/video caption等；

&emsp; 3）视觉属性，包含：画质增强，超分辨率，2D转3D，3D实时交互等；

&emsp; 4）多模态属性，视觉数据与语音数据、文本数据联合分析。


| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 4.what          | 4.1) 运动属性          | 2D Optical Flow (经典算法)      | Horn-Schunck光流法                                     | opencv.CalcOpticalFlowHS                                                                                    | 张熙瑞     |
|                 |                    |                             | Lucas-Kanade光流法                                     | cv::optflow::calcOpticalFlowSparseToDense()                                                                 | 张熙瑞     |
|                 |                    |                             | Block-Matching光流法                                   | opencv.calcOpticalFlowBM                                                                                    | 张熙瑞     |
|                 |                    |                             | Dual-TVL1                                           | https://docs.opencv.org/4.5.5/dc/d4d/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html                         | 张熙瑞     |
|                 |                    |                             | DeepFlow-v2                                         | http://lear.inrialpes.fr/src/deepflow/                                                                      | 张熙瑞     |
|                 |                    |                             | Global Patch Collider                               | https://docs.opencv.org/4.5.5/d8/dc5/sparse__matching__gpc_8hpp.html                                        | 张熙瑞     |
|                 |                    | 2D Optical Flow (深度学习)      | RAFT (ECCV 2020 best paper)                         | https://github.com/princeton-vl/RAFT                                                                        | 张熙瑞     |
|                 |                    |                             | FlowNet2.0                                          | https://github.com/NVIDIA/flownet2-pytorch                                                                  | 张熙瑞     |
|                 |                    |                             | NVIDIA SDK                                          | https://developer.nvidia.com/opticalflow-sdk                                                                | 张熙瑞     |
|                 |                    | 3D Scene Flow               | FlowNet3D                                           | https://github.com/xingyul/flownet3d                                                                        | 张熙瑞     |
|                 |                    |                             | Just Go with the Flow                               | https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation                     | 张熙瑞     |
|                 |                    |                             | MotionNet                                           | https://www.merl.com/research/?research=license-request&sw=MotionNet                                        | 张熙瑞     |
|                 |                    |                             | 2D-3D Expansion                                     | https://github.com/gengshan-y/expansion                                                                     | 张熙瑞     |
|                 | 4.2) 语义属性          | 图像检索识别                      | PP-Lite-Shitu                                       | https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/deploy/lite_shitu                               | 洪力      |
|                 |                    |                             | PP-LCNetV2                                          | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 | 洪力      |
|                 |                    | 视频动作识别                      | CTR-GCN                                             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/ctrgcn.md         | 洪力      |
|                 |                    |                             | ST-GCN                                              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md          | 洪力      |
|                 |                    |                             | AGCN                                                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/agcn.md           | 洪力      |
|                 |                    | Image Caption               | COCO Caption                                        | https://github.com/tylin/coco-caption                                                                       | 王庆忠     |
|                 |                    |                             | Im2Text                                             | https://www.cs.virginia.edu/~vicente/sbucaptions/                                                           | 王庆忠     |
|                 |                    | Video Caption               | ActivityNet                                         | http://activity-net.org/challenges/2017/captioning.html                                                     | 王庆忠     |
|                 | 4.3) 视觉属性          | 画质增强                        | Space-Time-Aware Multi-Resolution Video Enhancement | https://github.com/alterzero/STARnet                                                                        | 卢飞翔     |
|                 |                    | 图像/视频去噪                     | FastDVDnet                                          | https://github.com/m-tassano/fastdvdnet                                                                     | 卢飞翔     |
|                 |                    | 超分辨率                        | Super Resolution                                    |                                                                                                             | 卢飞翔     |
|                 |                    | 图像填补                        | Inpainting                                          |                                                                                                             | 卢飞翔     |
|                 |                    | 2D转3D                       | NeRF                                                |                                                                                                             | 卢飞翔     |
|                 |                    | 3D Visualization            | Maya                                                |                                                                                                             | 卢飞翔     |
|                 |                    |                             | Unity                                               |                                                                                                             | 卢飞翔     |
|                 |                    |                             | Unreal                                              |                                                                                                             | 卢飞翔     |
|                 | 4.4) 多模态属性         | 文本+视觉                       | VideoBERT                                           |                                                                                                             | 王庆忠     |
|                 |                    |                             | VisualBERT                                          |                                                                                                             | 王庆忠     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |

## 5. why

&emsp; “why”模块重点分析影响运动表现的因素，并尝试预测伤病的可能性、比赛成绩等：

&emsp; 1）采集生理、心理、体能相关数据，并与运动表现进行关联性分析；

&emsp; 2）从生物力学的角度，对动作细节进行纠正；

&emsp; 3）从内负荷、外负荷的角度，在确保训练强度的情况下，尽可能减少伤病发生的可能性。


| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 5.why           | 5.1) 分析            | 技术、生理、心理、体能                 |                                                     |                                                                                                             | 卢飞翔     |
|                 | 5.2) 推理            | 生物力学                        |                                                     |                                                                                                             | 卢飞翔     |
|                 | 5.3) 预测            | 内负荷、外负荷                     |                                                     |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 6. how

&emsp; “how”模块重点分析影响“AI+Sports”技术落地的因素：

&emsp; 1）费用，取决于数据标注数量和网络训练需要的GPU费用；

&emsp; 2）人力，重新训练模型所需的人力数量；

&emsp; 3）时间，配置、测试、重训练、重开发等所需要的时间。


| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 6.how           | 6.1) much          | 经费                          |                                                     |                                                                                                             | 卢飞翔     |
|                 | 6.2) many          | 人力                          |                                                     |                                                                                                             | 卢飞翔     |
|                 | 6.3) long          | 时间                          |                                                     |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 7. data

&emsp; “data”模块重点梳理生成训练数据的6种主流方式：

&emsp; 1）人工标注：已标注的公开数据集，用于网络训练；

&emsp; 2）迁移学习：未标注的大量数据，做非监督学习和迁移学习；

&emsp; 3）合成数据：2D图像直接编辑，copy-paste的方式合成训练数据；

&emsp; 4）合成数据：3D模型渲染生成2D数据以及标注信息；

&emsp; 5）合成数据：3D模型部件指导的2D图像编辑；

&emsp; 6）合成数据：GAN系列网络模型合成训练数据。



| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 7.data          | 7.1) 已标注的数据集       |                             |                                                     |                                                                                                             | 王庆忠     |
|                 | 7.2) 未标注的数据集       |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 7.3) 2D Copy-Paste |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 7.4) 3D Rendering  |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 7.5) 3D-2D Editing |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 7.6) GAN           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 8. benchmark

&emsp; “benchmark”模块将构建第一个体育类的benchmark，尽可能让所有算法在一个数据集上进行评测，特点是小而精，包含以下信息：

&emsp; 1）when：时域信息标注，回合起止节点；

&emsp; 2）where：2D/3D检测，2D分割，2D跟踪，2D/3D骨架；

&emsp; 3）who：人员分类，姓名；

&emsp; 4）what：运动，语义，视觉信息。

| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 8.benchmark     | 8.1) 训练数据集         |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 8.2) 测试数据集         |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 8.3) 评估脚本          |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |



## 9. tools

&emsp; 面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 9.tools         | 9.1) 标注工具          |                             |                                                     |                                                                                                             | 张孟希     |
|                 | 9.2) 检测工具          |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 9.3) 识别工具          |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 9.4) 深度图生成工具       |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 |                    |                             |                                                     |                                                                                                             |         |
|                 |                    |                             |                                                     |                                                                                                             |         |


## 10. applications

&emsp; 涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。

| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 10.applications | 10.1) 足球           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.2) 跳水           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.3) 乒乓球          |                             |                                                     |                                                                                                             | 张孟希     |
|                 | 10.4) 花样滑冰         |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.5) 健身           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.6) 篮球           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.7) 蹦床           |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.8) 大跳台          |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.9) 速度滑冰         |                             |                                                     |                                                                                                             | 卢飞翔     |
|                 | 10.10) 跑步          |                             |                                                     |                                                                                                             | 卢飞翔     |


# 合作伙伴
- 国家队
- 央视
- 国家体育总局体育科学研究所
- 高校：北京大学，北京航空航天大学，南京大学，大连理工大学，上海科技大学，厦门大学
- 体育类商业公司
- 世界冠军运动员、教练等

# 百度开发团队
- 百度研究院 机器人与自动驾驶实验室（RAL）
- 百度研究院 大数据实验室（BDL）
- 百度深度学习技术平台部（PaddlePaddle）
- 百度ACG产业创新业务部




