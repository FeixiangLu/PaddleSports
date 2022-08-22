# 使用Real-ESRGAN恢复低分辨率的足球场景照片

## 1. Real-ESRGAN简介

- [Real-ESRGAN](https://arxiv.org/abs/2107.10833)采用纯合成数据对ESRGAN朝着真实复原应用进行扩展，得到了本项目所提的Real-ESRGAN，这是ESRAGN、EDVR等超分领域里程碑式论文作者的又一力作
- Real-ESRGAN构建了一个**高阶退化建模过程**以更好的模拟复杂的真实退化。在合成过程中，同时还考虑的重建的ringing, overshoot伪影等问题

### 1.1 高阶退化建模

- 经典的退化模型仅仅包含固定数量的基本退化，这种退化可以视作**一阶退化**
- 然而，**实际生活中的退化过程非常多样性**，通常包含不同的处理，比如成像系统、图像编辑、网络传输等等。当我们想要对网络下载的低分辨率图像进行重建时，难度就很大了，例如会遇到以下问题：
    - 原始图像可能由多年前的手机拍摄所得，包含严重的退化问题
    - 当图像通过锐化软件编辑处理后又会引入overshoot以及模糊伪影等问题
    - 当图像经过网络传输后又会进一步引入不可预料的压缩噪声等

- 上述的这些操作都会使得退化变得很复杂，为缓解上述问题，Real-ESRGAN提出了**高阶退化模型**：它包含多个重复退化的过程，其中，每个阶段采用相同的退化处理但具有不同的退化超参，流程如下图所示（**注：高阶退化模型仍然不完美，无法覆盖真实世界的整个退化空间**。相反，它通过修改数据合成过程对已有盲图像超分的退化边界进行扩展）：

![](https://ai-studio-static-online.cdn.bcebos.com/756e5991803b45079ab0ef7362a4fb844af89dce7ca346c4993a5aa8f65acaec)

### 1.2 Real-ESRGAN效果

- 将Real-ESRGAN的torch权重转为paddle的权重，对手上有的**真实的低分辨率数据**（行车记录仪影像、300m高度飞行的DJI无人机影像）进行超分重建，结果如下图所示：

| 低分辨率影像 | Real-ESRGAN超分 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/060ea2ff925e444f877643f8b32b67590cd08115247b460eae785d0a0547c2d7)|![](https://ai-studio-static-online.cdn.bcebos.com/c51719e8a80148cdbc7edf4452a4e391749b3533503a4eb0943afcac99defeac) |
| ![](https://ai-studio-static-online.cdn.bcebos.com/7e7896a0865e4f54ad1d4cba637cc8fb557713f57c034d1d8a6dfb32f344bf86) | ![](https://ai-studio-static-online.cdn.bcebos.com/82af34f06cfe42b3a86bb334d3fb758687385c47403743bd837ef9320e7c63f3)|

### 1.3 对足球场景超分的不足

- Real-ESRGAN对包浆的自然影像的处理效果很好，但是对足球场景的图像通常会有一些**失真**的效果，例如**虚化的背景球员恢复出来扭曲的效果、球员的肢体皮肤出现不真实的伪影等**，选取了一些百度图片上的低分辨率足球场景图像块测试，如下所示：

| 低分辨图像 | Real-ESRGAN超分|
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/4454dbc0f9234f4a9d579d5f3ead4a8578d4e1b953ea44fb975e38600dfbca32)| ![](https://ai-studio-static-online.cdn.bcebos.com/e8f25869ad364d78b6ba7c87901d5e52ee0dd7c5fdee442e9d1a5bd6c512564f) |
| ![](https://ai-studio-static-online.cdn.bcebos.com/3c72f38ace444b7f9365bfc630e2e99effc005e4e39d4328b98e95cf57ec85b9)| ![](https://ai-studio-static-online.cdn.bcebos.com/5c82f0b08a12460e8f4c6bbb9d4842e96bf50f32ff0b4f69a6d53f30fa86dff2)|

- 由于Real-ESRGAN训练时没有针对足球场景的数据进行训练，所以本项目**使用paddle版本的Real-ESRGAN以及2022年欧冠决赛的高清视频的关键帧训练**，以期缓解此种现象
