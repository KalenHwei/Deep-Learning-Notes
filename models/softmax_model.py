import torch.nn as nn
import torch

class softmax_network(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=10, num_hiddens=2048):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.linear1 = nn.Linear(num_inputs, num_hiddens) # 784 -> 2048
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens, num_outputs) # 2048 -> 10
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, self.num_inputs) # 输入为[256,1,64,64]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    net = softmax_network()
    x = torch.randn(256, 1, 28, 28)
    output = net(x) 
    print(output.shape)# [256, 10] 是个logits