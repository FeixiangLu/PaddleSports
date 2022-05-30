# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1.整体采用“5W1H”的结构，即：when，where，who，what，why，how。系统梳理人工智能技术在体育行业的应用。

2.深度模型：精度、速度、集成度三个核心指标进行评测。

3.数据集：除了各个已有的公开数据集来评测深度模型的性能外，首次推出SportsBenchmark，能够用一个数据集来评测所有算法模型。

4.AI技术不仅是深度学习，同时整理了经典3D建模、SLAM、机器学习，以及硬件集成开发等工作，力图实现“百花齐放”。


| 任务      | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          |
|---------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|
| 1.when  | 1.1) 视频（时域）    | 视频分类（是什么体育项目）              | PP-TSM              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md         |
|         |                |                            | PP-TimeSformer      | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-timesformer.md |
|         |                |                            | SlowFast            | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md       |
|         |                |                            | AttentionLSTM       | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/attention_lstm.md |
|         |                |                            | MoViNet             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/movinet.md        |
|         |                | 片段切割（起始点，终止点）              | BMN                 | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md           |
|         |                | 动作识别（每一帧属于什么动作）            | MS-TCN              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/mstcn.md         |
|         |                |                            | ASRF                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/asrf.md          |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
| 2.where | 2.1) 2D检测      | 一阶段通用目标检测                  | PP-YOLOE            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe                            |
|         |                |                            | PP-PicoDet          | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet                            |
|         |                | 二阶段通用目标检测                  | Faster-RCNN         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        |
|         |                | 人体检测分析                     | PP-Human            | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman                             |
|         |                |                            | PP-Pedestrian       | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/pedestrian                         |
|         |                | 水花/足球/篮球等小目标检测             | FPN                 | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn                        |
|         |                |                            |                     |                                                                                                             |
|         | 2.2) 2D分割      | 前景对象/背景分割                  | Mask-RCNN           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mask_rcnn                          |
|         |                |                            | SOLOv2              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/solov2                             |
|         |                |                            | PP-LiteSeg          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/pp_liteseg                               |
|         |                |                            | DeepLabV3P          | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p                               |
|         |                | 交互式分割                      | EISeg               | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/EISeg                                            |
|         |                | 人体分割                       | PP-HumanSeg         | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/contrib/PP-HumanSeg                              |
|         |                | 人体毛发级精准分割                  | Matting             | https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/Matting                                          |
|         |                |                            | Human Matting       | https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/Matting/model/human_matting.py                   |
|         |                | 视频目标分割                     | CFBI                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/segmentation/cfbi.md          |
|         |                |                            | MA-Net              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/EIVideo/EIVideo/docs/zh-CN/manet.md   |
|         |                | 视频运动物体分割                   | Motion Segmentation |                                                                                                             |
|         |                | 视频人体分割 video matting       | BackgroundMattingV2 | https://github.com/PeterL1n/BackgroundMattingV2                                                             |
|         |                |                            |                     |                                                                                                             |
|         | 2.3) 2D跟踪      | 人体跟踪                       | ByteTrack           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mot/bytetrack                      |
|         |                | 运动轨迹                       | PP-Tracking         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pptracking                          |
|         |                |                            |                     |                                                                                                             |
|         | 2.4) 2D骨骼      | Top-Down                   | PP-TinyPose         | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose                 |
|         |                |                            | HR-Net              | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint                           |
|         |                | Bottom-Up                  | OpenPose            | https://github.com/CMU-Perceptual-Computing-Lab/openpose                                                    |
|         |                |                            | MoveNet             | https://github.com/fire717/movenet.pytorch                                                                  |
|         |                |                            |                     |                                                                                                             |
|         | 2.5) 3D骨骼      | 单目                         | Position-based      |                                                                                                             |
|         |                |                            | Angle-based         |                                                                                                             |
|         |                |                            | 2D + Depth-based    |                                                                                                             |
|         |                |                            | 2D + IK             |                                                                                                             |
|         |                | 多目                         | Calibration         |                                                                                                             |
|         |                |                            | Fusion              |                                                                                                             |
|         |                | 深度相机                       | Kinect 3D Tracking  | https://docs.microsoft.com/zh-cn/azure/Kinect-dk/get-body-tracking-results                                  |
|         |                |                            |                     |                                                                                                             |
|         | 2.6) 3D稠密关键点   | 2D-3D Dense Correspondence | DensePose           | https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose                                 |
|         |                |                            |                     |                                                                                                             |
|         | 2.7) 3D人体重建    | Template Model             | SMPL                | https://smpl.is.tue.mpg.de/                                                                                 |
|         |                |                            | VIBE                | https://github.com/mkocabas/VIBE                                                                            |
|         |                |                            | PyMaf               | https://github.com/HongwenZhang/PyMAF                                                                       |
|         |                |                            |                     |                                                                                                             |
|         | 2.8) SLAM      | 静态                         | 单目 ORB-SLAM...      | https://github.com/UZ-SLAMLab/ORB_SLAM3                                                                     |
|         |                |                            | 深度 KinectFusion...  | https://github.com/victorprad/InfiniTAM                                                                     |
|         |                |                            | 激光 LOAM             | https://github.com/RobustFieldAutonomyLab/LeGO-LOAM                                                         |
|         |                | 动态                         | DynamicFusion       | https://github.com/mihaibujanca/dynamicfusion                                                               |
|         |                |                            | DynSLAM             | https://github.com/AndreiBarsan/DynSLAM                                                                     |
|         |                |                            |                     |                                                                                                             |
|         | 2.9) 相机6-DoF定位 | 内参                         | 张氏标定法               |                                                                                                             |
|         |                | 外参                         | 单张图像 PNP            |                                                                                                             |
|         |                |                            | 多张图像 SfM, SLAM      |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
| 3.who   | 3.1) 识别        | 人脸检测                       | BlazeFace           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/face_detection                     |
|         |                | 人脸识别                       | Dlib                | http://dlib.net/                                                                                            |
|         |                | 基于人体的运动员识别                 |                     |                                                                                                             |
|         |                | 运动员Re-ID                   | MultiSports         | https://github.com/MCG-NJU/MultiSports                                                                      |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
| 4.what  | 4.1) 运动属性      | 2D optical flow            |                     |                                                                                                             |
|         |                | 3D scene flow              |                     |                                                                                                             |
|         | 4.2) 语义属性      | 图像检索识别                     | PP-Lite-Shitu       | https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/deploy/lite_shitu                               |
|         |                |                            | PP-LCNetV2          | https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md                 |
|         |                | 视频动作识别                     | CTR-GCN             | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/ctrgcn.md         |
|         |                |                            | ST-GCN              | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md          |
|         |                |                            | AGCN                | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/agcn.md           |
|         |                | image caption              | COCO Caption        | https://github.com/tylin/coco-caption                                                                       |
|         |                |                            | Im2Text             | https://www.cs.virginia.edu/~vicente/sbucaptions/                                                           |
|         |                | video caption              | ActivityNet         | http://activity-net.org/challenges/2017/captioning.html                                                     |
|         | 4.3) 视觉属性      | 3D Visualization           | Maya                |                                                                                                             |
|         |                |                            | Unity               |                                                                                                             |
|         |                |                            | Unreal              |                                                                                                             |
|         | 4.4) 多模态       | 文本+语音                      | VideoBERT           |                                                                                                             |
|         |                |                            | VisualBERT          |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
| 5.why   | 5.1) 分析        | 技术、生理、心理、体能                |                     |                                                                                                             |
|         | 5.2) 推理        | 生物力学                       |                     |                                                                                                             |
|         | 5.3) 预测        | 内负荷、外负荷                    |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
| 6.how   | 6.1) much      | 经费                         |                     |                                                                                                             |
|         | 6.2) many      | 人力                         |                     |                                                                                                             |
|         | 6.3) long      | 时间                         |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |
|         |                |                            |                     |                                                                                                             |


# 分模块介绍
该部分详细介绍“5W1H”各个模块的内容。

## 1.sports_when

## 2.sports_where

## 3.sports_who

## 4.sports_what

## 5.sports_why

## 6.sports_how

# 开发团队
