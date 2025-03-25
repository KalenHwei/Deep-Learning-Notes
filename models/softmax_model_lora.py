import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 

from lora.lora import LoRALinear

class softmax_network_lora(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=10, num_hiddens=2048, lora_r=4, lora_alpha=1.0):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.linear1 = LoRALinear(num_inputs, num_hiddens, r=lora_r, alpha=lora_alpha)  # 784 -> 2048
        self.relu = nn.ReLU()
        self.linear2 = LoRALinear(num_hiddens, num_outputs, r=lora_r, alpha=lora_alpha)  # 2048 -> 10
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, self.num_inputs)  # 输入形状调整
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
