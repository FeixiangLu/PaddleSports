# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[when（什么时间）](#1-sportswhen)，[where（什么位置）](#2-sportswhere)，[who（是谁）](#3-sportswho)，[what（发生了什么）](#4-sportswhat)，[why（为什么）](#5-sportswhy)，[how（怎么样）](#6-sportshow)。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. AI模型：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [数据集](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[SportsBenchmark](#8-sportsbenchmark)，力争能够用一个数据集来评测所有算法模型。

4. [工具集](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [应用](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。



# sports_where

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

