# 深度学习仓库
## Introduction
本仓库基于训练框架逐步拓展，基于不同Backbone从零实现预训练、LoRA PEFT到RLHF PPO、GRPO等算法。

## Deployment
### 环境配置：
1 `conda create -n <your_env_name> python=3.11`

2 `pip install torch torchvision transformer peft`

### 使用说明
`train_with_finetune.py`支持了基于分类任务的预训练与LoRA微调，其中LoRA无调用peft封装。基于此，`train_with_finetune_transformer.py`支持了transformer encoder。模型存放在`models`中。如需实验，请通过`download_mnist.py`下载简单的十分类数据集。
