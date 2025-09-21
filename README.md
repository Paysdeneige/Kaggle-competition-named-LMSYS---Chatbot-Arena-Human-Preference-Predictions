# LMSYS Chatbot Arena Human Preference Predictions

A deep learning project for predicting human preferences in chatbot conversations. Based on the DeBERTa-v3 model, this project uses a dual-tower architecture to compare the quality of responses from two chatbots and predict which model's output a human would prefer.

## Project Overview

In the LMSYS Chatbot Arena, users are presented with responses from two different AI models and can select the one they prefer. This project aims to train a model to automatically predict this human preference, which can be used to:
- Automatically evaluate the quality of AI model responses
- Reduce manual annotation costs
- Provide feedback for model improvement

## Core Features

- 🏗️ **Dual-tower architecture**: Encode the responses of two models separately and then compare them
- 🔄 **Flexible training mode**: Supports classification (3 categories) and ranking (regression) tasks
- 📊 **Multi-metric evaluation**: Includes precision, recall, F1-score and log_loss
- 🚀 **Distributed training**: Supports multi-GPU training
- ⚡ **Mixed precision**: Supports FP16 training acceleration

## Project Structure

```
├── models(1).py # Model architecture definition
├── utils_datasets.py # Data processing and dataset classes
├── train.py # Training script
├── run.sh # Run script
└── README.md # Project Description
```

## Environment Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- CUDA (for GPU training)

## Installation Dependencies

```bash
pip install torch transformers pandas scikit-learn numpy scipy openpyxl accelerate
```

## Data Format

Training data should be in CSV format, containing the following columns:
- `prompt`: Prompt for user input
- `response_a`: Response from Model A
- `response_b`: Response from Model B
- `winner_model_a`: Winning label for Model A (0 or 1)
- `winner_model_b`: Winning label for Model B (0 or 1)
- `winner_tie`: Tie label (0 or 1)

## Usage

### 1. Prepare the data
Save the training data as a `train.csv` file in the project root directory.

### 2. Configure the model
In `train.py`, modify the following parameters:
```python
model_name = './deberta-v3-base' # Model path
save_dir = './output/deberta_v3_base/' # Save directory
MAX_LEN = 1024 # Maximum sequence length
if_use_rank = False # Whether to use ranking mode
```

### 3. Start training

#### Single-GPU training
```bash
python train.py
```

#### Multi-GPU training
```bash
bash run.sh
```

Or use accelerate:
```bash
accelerate launch train.py
```

## Model Architecture

### Dual-Tower Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             输入层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Prompt + Model_A_Response  │  Prompt + Model_B_Response                    │
│  (文本序列)                 │  (文本序列)                                   │
└─────────────────────┬─────────────────────┬─────────────────────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             编码层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  DeBERTa-v3 Encoder A      │  DeBERTa-v3 Encoder B                         │
│  (共享权重)                │  (共享权重)                                    │
│  ┌─────────────────────┐   │  ┌─────────────────────┐                      │
│  │ Token Embedding     │   │  │ Token Embedding     │                      │
│  │ Position Embedding  │   │  │ Position Embedding  │                      │
│  │ Type Embedding      │   │  │ Type Embedding      │                      │
│  └─────────────────────┘   │  └─────────────────────┘                      │
│  ┌─────────────────────┐   │  ┌─────────────────────┐                      │
│  │ Transformer Layers  │   │  │ Transformer Layers  │                      │
│  │ (12 layers)         │   │  │ (12 layers)         │                      │
│  │ - Self-Attention    │   │  │ - Self-Attention    │                      │
│  │ - FFN               │   │  │ - FFN               │                      │
│  │ - Layer Norm        │   │  │ - Layer Norm        │                      │
│  └─────────────────────┘   │  └─────────────────────┘                      │
│  ┌─────────────────────┐   │  ┌─────────────────────┐                      │
│  │ Hidden States A     │   │  │ Hidden States B     │                      │
│  │ [batch, seq_len,    │   │  │ [batch, seq_len,    │                      │
│  │  hidden_size=768]   │   │  │  hidden_size=768]   │                      │
│  └─────────────────────┘   │  └─────────────────────┘                      │
└─────────────────────┬─────────────────────┬─────────────────────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             池化层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  MeanPooling A             │  MeanPooling B                                 │
│  ┌─────────────────────┐   │  ┌─────────────────────┐                      │
│  │ Attention Mask      │   │  │ Attention Mask      │                      │
│  │ Weighted Average    │   │  │ Weighted Average    │                      │
│  └─────────────────────┘   │  └─────────────────────┘                      │
│  ┌─────────────────────┐   │  ┌─────────────────────┐                      │
│  │ Sentence Embedding A│   │  │ Sentence Embedding B│                      │
│  │ [batch, 768]        │   │  │ [batch, 768]        │                      │
│  └─────────────────────┘   │  └─────────────────────┘                      │
└─────────────────────┬─────────────────────┬─────────────────────────────────┘
                      │                     │
                      └─────────┬───────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             融合层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Concatenation                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ [Sentence_A, Sentence_B]                                               │ │
│  │ [batch, 1536]                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             分类层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Fully Connected Layer                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Linear(1536 → 3)                                                       │ │
│  │ Weight: Normal(0, 0.02)                                                │ │
│  │ Bias: 0                                                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             输出层                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Logits [batch, 3]                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ [score_model_a, score_model_b, score_tie]                              │ │
│  │ 经过Softmax后得到概率分布                                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘                                                        ┘
```

### Main Components

1. **CustomModel**: Main classification model

- Input: Responses from two models

- Output: 3-category classification results (model_a wins/model_b wins/draw)

2. **CustomModelRank**: Ranking model

- Input: Responses from two models

- Output: Continuous score for ranking

3. **MeanPooling**: Average pooling layer

- Aggregates sequence-level features into sentence-level representations

## Training Parameters

- **Learning Rate**: 5e-5
- **Number of Training Episodes**: 5 epochs
- **Batch Size**: 2 (per device)
- **Evaluation Strategy**: Evaluate every 2000 steps
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss (classification) / MSELoss (ranking)

## Evaluation Metrics

- **Log Loss**: Main optimization objective
- **Precision**: Precision
- **Recall**: Recall rate
- **F1-Score**: F1 score

## Output files

After training, the model and tokenizer will be saved in the specified output directory:

```
output/deberta_v3_base/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

## Custom configuration

### Modify model architecture
In `models(1).py`, you can:
- Adjust hidden layer dimensions
- Add additional layers (such as BiLSTM)
- Modify pooling strategy

### Modify data processing
In `utils_datasets.py`, you can:
- Adjust maximum sequence length
- Modify label mapping
- Add data augmentation

### Modify training strategy
In `train.py`, you can:
- Adjust learning rate and number of training rounds
- Modify batch size
- Adding New Evaluation Metrics

## Troubleshooting

### Common Problems

1. **CUDA Out of Memory**
- Reduce batch size
- Reduce maximum sequence length
- Use gradient accumulation

2. **Slow Training**
- Enable mixed-precision training (FP16)
- Use more GPUs
- Reduce evaluation frequency

3. **Model Not Converging**
- Adjust learning rate
- Increase warmup steps
- Check data quality

## Contribution Guidelines

Issues and pull requests are welcome to contribute to the project!

## License

This project is licensed under the MIT License.

## Acknowledgements

- [LMSYS Chatbot Arena](https://chat.lmsys.org/) 提供数据集
- [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa) 提供基础模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) 提供训练框架
