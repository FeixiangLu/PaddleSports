# PaddleSports

# 框架介绍
PaddleSports是飞桨面向体育场景的端到端开发套件，实现人工智能技术与体育行业的深度融合，目标打造“AI+Sports”的标杆案例集。PaddleSports的特色如下：

1. 整体采用“5W1H”的产品架构，即：[when（什么时间）](#1-sportswhen)，[where（什么位置）](#2-sportswhere)，[who（是谁）](#3-sportswho)，[what（发生了什么）](#4-sportswhat)，[why（为什么）](#5-sportswhy)，[how（怎么样）](#6-sportshow)。系统梳理人工智能技术在体育行业的研究、应用、落地。

2. AI模型：从精度、速度、集成度三个维度进行性能评测。AI技术不仅是深度学习，同时整理了经典3D建模，SLAM，机器学习，以及硬件集成开发等工作，目标打造软硬一体的“AI+Sports”开发套件。

3. [数据集](#7-data)：除了各个已有的公开数据集来评测深度模型的性能外，将首次推出[SportsBenchmark](#8-sportsbenchmark)，力争能够用一个数据集来评测所有算法模型。

4. [工具集](#9-tools)：面向体育场景的工具集，比如标注工具、检测工具、识别工具等，具有All-in-One，AutoRun的特点。

5. [应用](#10-applications)：涵盖足球、跳水、乒乓球、花样滑冰、健身、篮球、蹦床、大跳台、速度滑冰、跑步等热门的体育运动。


# sports_when
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







