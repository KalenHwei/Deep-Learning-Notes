import torch
import torchvision
import torch.nn as nn
import os
from torchvision import transforms
from typing import Any, Dict, List, Tuple


import download_mnist
from models.softmax_model import softmax_network
from models.softmax_model_lora import softmax_network_lora

def evaluate_accuracy(data_iter, net, device):
    """在指定设备上评估模型在数据集上的准确率"""
    net.eval()  # 设置为评估模式
    correct, total = 0, 0
    with torch.no_grad():
        for train_images, train_labels in data_iter:
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            predict_labels = net(train_images)
            _, predicted = predict_labels.max(1)
            total += train_labels.size(0)
            correct += (predicted == train_labels).sum().item()

    return correct / total

def train(train_iter,
           net, 
           loss, 
           num_epochs, 
           optimizer,
           device=torch.device("cuda" if torch.cuda.is_available() else "mps"),
           is_finetune: bool = False,
           checkpoint_path: str = None):

    net.to(device)
    start_epoch = 0

    if is_finetune == True and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"加载了来自第 {checkpoint['epoch']+1} 个epoch的checkpoint，继续微调训练...")
    else:
        print("非微调模式，从头开始训练。")

    for epoch in range(start_epoch, num_epochs+start_epoch):
        net.train()
        train_loss, train_acc, total_correct, total_samples = 0.0, 0.0, 0, 0

        for train_images, train_labels in train_iter:

            train_images, train_labels = train_images.to(device), train_labels.to(device)
            optimizer.zero_grad() # 优化器梯度清零

            predict_labels = net(train_images) # 前向传播，得到的是shape为(batch_size, 10)的tensor，即每个样本对应10个类别的概率
            l = loss(predict_labels, train_labels) # 计算损失，通过交叉熵损失函数计算真实和预测矩阵得到
            l.backward() # 反向传播
            optimizer.step() # 更新参数

            total_train_loss = train_loss + l.item() * train_labels.size(0) # 累加损失
            _, predict_items = predict_labels.max(1) # 返回每行最大值的索引，即每个样本预测的标签
            total_correct += (predict_items == train_labels).sum().item() # 计算预测正确的样本数
            total_samples += train_labels.size(0) # 累计当前批次的样本数，以便记录整个训练过程中处理的样本总数

        train_loss = total_train_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate_accuracy(test_iter, net, device)
        
        print(f"Epoch {epoch+1:2d}: Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

        # 每个epoch结束后保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if is_finetune == False:
            if not os.path.exists("checkpoints/pretrain"):
                os.makedirs("checkpoints/pretrain")
            torch.save(checkpoint, "checkpoints/pretrain/checkpoint.pth")
            if epoch == num_epochs-1:
                print(f"已保存预训练第 {epoch+1} 个epoch 的checkpoint到: checkpoints/pretrain")
        else:
            if not os.path.exists("checkpoints/finetune"):
                os.makedirs("checkpoints/finetune")
            torch.save(checkpoint, "checkpoints/finetune/checkpoint.pth")
            if epoch == num_epochs + start_epoch -1:
                print(f"已保存微调第 {epoch+1} 个epoch 的checkpoint到: checkpoints/finetune")



if __name__ == '__main__':
    batch_size = 256
    num_epochs = 10
    lr = 10e-2

    train_iter, test_iter = download_mnist.load_data_fashion_mnist(batch_size=batch_size)

    net = softmax_network()
    # net = softmax_network_lora(num_inputs=784, num_outputs=10, num_hiddens=2048, lora_r=4, lora_alpha=1.0)

    loss = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # train(train_iter, net, loss, num_epochs, optimizer)
    train(train_iter, net, loss, num_epochs, optimizer, is_finetune=True, checkpoint_path="checkpoints/pretrain/checkpoint.pth")