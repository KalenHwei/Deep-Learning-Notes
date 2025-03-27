import torch
import torchvision
import torch.nn as nn
import os
from torchvision import transforms
from typing import Any, Dict, List, Tuple

import download_mnist
from models.softmax_model import softmax_network
from models.softmax_model_lora import softmax_network_lora

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

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
           test_iter,
           net, 
           loss, 
           num_epochs, 
           optimizer,
           device=torch.device("cuda" if torch.cuda.is_available() else "mps"),
           is_finetune: str = "none", # "none", "lora", "lora_ppo"
           checkpoint_path: str = None,
           save_path: str = None):
    
    if not os.path.exists("checkpoints/pretrain"):
        os.makedirs("checkpoints/pretrain")      

    if not os.path.exists("checkpoints/finetune"):
        os.makedirs("checkpoints/finetune")

    net.to(device)
    start_epoch = 0
    best_test_acc = 0.0
    best_result_epoch = 0

    if is_finetune == True and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 微调阶段，主要关注的是模型权重的调整，而优化器的状态（例如动量、二阶矩等）只是辅助变量，用于加速和稳定训练过程
        # 因此，重新初始化优化器状态可以使微调过程从一个“干净”的状态开始，这通常不会对最终性能产生显著负面影响，反而能避免旧状态对新训练动态的干扰。
        # 也就是说，无需加载优化器状态，直接重新初始化优化器即可
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"加载了来自第 {checkpoint['epoch']+1} 个epoch的checkpoint，继续微调训练...")
    else:
        print("非微调模式，从头开始训练。")

    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, test_acc, total_correct, total_samples = 0.0, 0.0, 0.0, 0, 0

        for train_images, train_labels in train_iter:

            train_images, train_labels = train_images.to(device), train_labels.to(device)
            optimizer.zero_grad() # 优化器梯度清零

            predict_logits = net(train_images) # 前向传播，得到的是shape为(batch_size, 10)的tensor，即每个样本对应10个类别的概率
            l = loss(predict_logits, train_labels) # 计算损失，通过交叉熵损失函数计算真实和预测矩阵得到
            l.backward() # 反向传播
            optimizer.step() # 更新参数

            total_train_loss = train_loss + l.item() * train_labels.size(0) # 累加损失
            _, predict_items = torch.softmax(predict_logits,dim=1).max(1) # 返回每行最大值的索引，即每个样本预测的标签
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

        if is_finetune == False: # 预训练
                     
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_result_epoch = epoch
                torch.save(checkpoint, save_path)

            if epoch == num_epochs - 1:
                total_params, trainable_params = count_parameters(net)
                print(f"总参数数量: {total_params}, 训练参数数量: {trainable_params}, 更新参数量为: {trainable_params/total_params:.4f}")
                print(f"已保存预训练第 {best_result_epoch+1} 个最佳结果的epoch的checkpoint到:{save_path}")
    
        else: # 微调

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_result_epoch = epoch
                torch.save(checkpoint, save_path)

            if epoch == num_epochs - 1:
                total_params, trainable_params = count_parameters(net)
                print(f"总参数数量: {total_params}, 训练参数数量: {trainable_params}, 更新参数量为: {trainable_params/total_params:.4f}")
                print(f"已保存微调第 {best_result_epoch+1} 个最佳结果的epoch的checkpoint到: {save_path}")

if __name__ == '__main__':
    batch_size = 256
    num_epochs = 10
    lr = 10e-2

    train_iter, test_iter = download_mnist.load_data_fashion_mnist(batch_size=batch_size)

    # 定义训练网络：预训练时r=0，alpha=0；微调时r=4，alpha=1.0
    #net = softmax_network_lora(num_inputs=784, num_outputs=10, num_hiddens=2048, lora_r=0, lora_alpha=0)
    net = softmax_network_lora(num_inputs=784, num_outputs=10, num_hiddens=2048, lora_r=4, lora_alpha=1.0)

    loss = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 定义训练：预训练选第一个，微调选第二个
    #train(train_iter=train_iter, test_iter=test_iter, net=net, loss=loss, num_epochs=10, optimizer=optimizer, save_path="checkpoints/pretrain/checkpoint.pth")
    train(train_iter=train_iter, test_iter=test_iter, net=net, loss=loss, num_epochs=10, optimizer=optimizer, is_finetune=True, checkpoint_path="checkpoints/pretrain/checkpoint.pth", save_path="checkpoints/finetune/checkpoint.pth")