# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1.整体采用“5W1H”的产品架构，即：when，where，who，what，why，how。系统梳理人工智能技术在体育行业的应用。

2.深度模型：从精度、速度、集成度三个维度进行指标评测。

3.数据集：除了各个已有的公开数据集来评测深度模型的性能外，首次推出SportsBenchmark，力争能够用一个数据集来评测所有算法模型。

4.AI技术不仅是深度学习，同时整理了经典3D建模、SLAM、机器学习，以及硬件集成开发等工作，打造软硬一体的AI系统。


# 分模块介绍
该部分详细介绍“5W1H”各个模块的内容。

## 1.sports_when
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


## 2.sports_where
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


## 3.sports_who
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 3.who           | 3.1) 识别        | 人脸检测                       | BlazeFace           | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/face_detection                     | 王成      |
|                 |                | 人脸识别                       | Dlib                | http://dlib.net/                                                                                            | 王成      |
|                 |                | 基于人体的运动员识别                 |                     |                                                                                                             | 王成      |
|                 |                | 运动员Re-ID                   | MultiSports         | https://github.com/MCG-NJU/MultiSports                                                                      | 王成      |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 4.sports_what
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

## 5.sports_why
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 5.why           | 5.1) 分析        | 技术、生理、心理、体能                |                     |                                                                                                             | 卢飞翔     |
|                 | 5.2) 推理        | 生物力学                       |                     |                                                                                                             | 卢飞翔     |
|                 | 5.3) 预测        | 内负荷、外负荷                    |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 6.sports_how
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 6.how           | 6.1) much      | 经费                         |                     |                                                                                                             | 卢飞翔     |
|                 | 6.2) many      | 人力                         |                     |                                                                                                             | 卢飞翔     |
|                 | 6.3) long      | 时间                         |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 7.data
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 7.data          | 7.1) 公开的数据集    |                            |                     |                                                                                                             | 王庆忠     |
|                 | 7.2) 自有的数据集    |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 7.3) 待构建的数据集   |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 8.tools
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 8.tools         | 8.1) 标注工具      |                            |                     |                                                                                                             | 张孟希     |
|                 | 8.2) 深度图生成工具   |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |


## 9.sports_benchmark
| 任务              | 技术方向           | 技术细分                       | 算法模型                | 链接                                                                                                          | 人力安排    |
|-----------------|----------------|----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------|---------|
| 9.benchmark     | 9.1) 训练数据集     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 9.2) 测试数据集     |                            |                     |                                                                                                             | 卢飞翔     |
|                 | 9.3) 评估脚本      |                            |                     |                                                                                                             | 卢飞翔     |
|                 |                |                            |                     |                                                                                                             |         |
|                 |                |                            |                     |                                                                                                             |         |



## 10.applications
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


# 开发团队
- 百度研究院 机器人与自动驾驶实验室（RAL）

- 百度研究院 大数据实验室（BDL）

- 深度学习技术平台部（PaddlePaddle）





