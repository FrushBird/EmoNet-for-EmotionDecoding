# EmoNet-for-MoodMission
A little project from Xiaoming
- 简介
  
- 数据集
  我们对参与实验的非人灵长类动物施加了24种不同的刺激，并用位于杏仁核32通道电极采集了神经电生理数据。
  这批数据已经过sorting，选取出比较明显的细胞

- 模型架构
  两步模型，第一部分是预训练的解码器，第二部分是ResNet18

- 工作流
  step1 预处理：
  对原始神经电生理数据sorting，筛选出有效细胞；
  从‘刺激开始’时间戳选取合适的时间片段切分并提取相应细胞的spike；
  对每个spike选取一定的滑动时间窗计算发放率，组成发放率矩阵；
  
  step2 VAE预训练解码器：
  
  step3 输入数据执行端到端的分类任务：
  
- paper在写
