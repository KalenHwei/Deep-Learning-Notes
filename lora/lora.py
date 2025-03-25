import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            r: LoRA 的低秩维度（建议较小的整数，如 4、8）
            alpha: 缩放因子
            bias: 是否使用偏置项
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        # 原始权重参数，初始化后冻结（LoRA 方法中不更新）
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        # 冻结原有参数
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 如果 r > 0，则定义 LoRA 参数
        if r > 0:
            # lora_A: shape=(r, in_features)
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            # lora_B: shape=(out_features, r)
            self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        # 初始化原始权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 标准全连接层计算（原始权重）
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # 计算 LoRA 部分：先 x 与 lora_A^T，再与 lora_B^T 相乘
            # 即 x * (lora_A^T) 得到 shape=(batch, r)
            # 再乘以 (lora_B^T) 得到 shape=(batch, out_features)
            lora_update = F.linear(x, self.lora_A, bias=None)
            lora_update = F.linear(lora_update, self.lora_B, bias=None)
            result = result + self.scaling * lora_update
        return result