from models.simple_transformer import TransformerEncoder
from models.softmax_model_lora import softmax_network_lora
import torch
import torch.nn as nn

class transformer_encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1, lora_r=0, lora_alpha=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout, lora_r=lora_r, lora_alpha=lora_alpha)
        self.classification_head = softmax_network_lora(num_inputs=embed_dim, num_outputs=10, num_hiddens=2048, lora_r=lora_r, lora_alpha=lora_alpha)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.reshape(batch_size, channels, -1) # 输入形状调整
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

if __name__ == '__main__':
    net = transformer_encoder(num_layers=3, embed_dim=784, num_heads=8, ff_dim=1536, dropout=0.1, lora_r=0, lora_alpha=0.0)
    x = torch.randn(256, 1, 28, 28)
    output = net(x)
    print(output.shape)
