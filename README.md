# LMSYS Chatbot Arena Human Preference Predictions

一个用于预测聊天机器人对话中人类偏好的深度学习项目。该项目基于DeBERTa-v3模型，采用双塔架构来比较两个聊天机器人的回复质量，预测人类会更偏好哪个模型的输出。

## 项目概述

在LMSYS Chatbot Arena中，用户会看到两个不同AI模型的回复，并选择他们更喜欢的那个。本项目旨在训练一个模型来自动预测这种人类偏好，可以用于：
- 自动评估AI模型回复质量
- 减少人工标注成本
- 为模型改进提供反馈

## 核心特性

- 🏗️ **双塔架构**: 分别编码两个模型的回复，然后进行比较
- 🔄 **灵活训练模式**: 支持分类(3类)和排序(回归)两种任务
- 📊 **多指标评估**: 包含precision、recall、F1-score和log_loss
- 🚀 **分布式训练**: 支持多GPU训练
- ⚡ **混合精度**: 支持FP16训练加速

## 项目结构

```
├── models(1).py          # 模型架构定义
├── utils_datasets.py     # 数据处理和数据集类
├── train.py             # 训练脚本
├── run.sh               # 运行脚本
└── README.md            # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- CUDA (用于GPU训练)

## 安装依赖

```bash
pip install torch transformers pandas scikit-learn numpy scipy openpyxl accelerate
```

## 数据格式

训练数据应为CSV格式，包含以下列：
- `prompt`: 用户输入的提示
- `response_a`: 模型A的回复
- `response_b`: 模型B的回复
- `winner_model_a`: 模型A获胜的标签(0或1)
- `winner_model_b`: 模型B获胜的标签(0或1)
- `winner_tie`: 平局的标签(0或1)

## 使用方法

### 1. 准备数据
将训练数据保存为`train.csv`文件，放在项目根目录下。

### 2. 配置模型
在`train.py`中修改以下参数：
```python
model_name = './deberta-v3-base'  # 模型路径
save_dir = './output/deberta_v3_base/'  # 保存路径
MAX_LEN = 1024  # 最大序列长度
if_use_rank = False  # 是否使用排序模式
```

### 3. 开始训练

#### 单GPU训练
```bash
python train.py
```

#### 多GPU训练
```bash
bash run.sh
```

或使用accelerate：
```bash
accelerate launch train.py
```

## 模型架构

### 双塔架构设计

```
Prompt + Model_A_Response ──┐
                           ├─> DeBERTa ──> MeanPooling ──┐
Prompt + Model_B_Response ──┘                             ├─> Concat ──> FC ──> Score
                                                          ┘
```

### 主要组件

1. **CustomModel**: 主要的分类模型
   - 输入：两个模型的回复
   - 输出：3类分类结果(model_a胜/model_b胜/平局)

2. **CustomModelRank**: 排序模型
   - 输入：两个模型的回复
   - 输出：连续分数用于排序

3. **MeanPooling**: 平均池化层
   - 将序列级别的特征聚合为句子级别的表示

## 训练参数

- **学习率**: 5e-5
- **训练轮数**: 5 epochs
- **批次大小**: 2 (per device)
- **评估策略**: 每2000步评估一次
- **优化器**: AdamW
- **损失函数**: CrossEntropyLoss (分类) / MSELoss (排序)

## 评估指标

- **Log Loss**: 主要优化目标
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数

## 输出文件

训练完成后，模型和tokenizer将保存在指定的输出目录中：
```
output/deberta_v3_base/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

## 自定义配置

### 修改模型架构
在`models(1).py`中可以：
- 调整隐藏层维度
- 添加额外的层(如BiLSTM)
- 修改池化策略

### 修改数据处理
在`utils_datasets.py`中可以：
- 调整最大序列长度
- 修改标签映射
- 添加数据增强

### 修改训练策略
在`train.py`中可以：
- 调整学习率和训练轮数
- 修改批次大小
- 添加新的评估指标

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批次大小
   - 减少最大序列长度
   - 使用梯度累积

2. **训练速度慢**
   - 启用混合精度训练(FP16)
   - 使用更多GPU
   - 减少评估频率

3. **模型不收敛**
   - 调整学习率
   - 增加warmup步数
   - 检查数据质量

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。

## 致谢

- [LMSYS Chatbot Arena](https://chat.lmsys.org/) 提供数据集
- [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa) 提供基础模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) 提供训练框架
