#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:36:04 2024

@author: btian
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from tqdm import tqdm
import torch.multiprocessing as mp
import os

# Function to normalize the tensor to range [0, 1] using fixed min and max values
def normalize_tensor(tensor, min_val=140.0, max_val=280.0):
    return (tensor - min_val) / (max_val - min_val)

# 加载数据
Xtrain = torch.load('/scratch/htc/btian/NNs_test/train_samples.pt').float()
Ytrain = torch.load('/scratch/htc/btian/NNs_test/ground_truth_train.pt').float()

# Normalize Xtrain using fixed min and max values
Xtrain = normalize_tensor(Xtrain)
print(Xtrain)
# 创建数据集和数据加载器
dataset = TensorDataset(Xtrain, Ytrain)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(Xtrain.shape[1], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, Ytrain.shape[1])

    def forward(self, x):
        print(x.dtype)
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = torch.sin(self.fc3(x))
        x = torch.sin(self.fc4(x))
        x = torch.sin(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))  # 将输出限制在(0, 1)
        return x

# Helper function to move state_dict to CPU
def state_dict_to_cpu(state_dict):
    return {key: value.cpu() for key, value in state_dict.items()}

# 训练模型函数
def train_model(rank, train_loader, val_loader, epochs, device, return_dict, save_path):
    print(f"Rank {rank}: Using device: {device}")
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accelerator = Accelerator()

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    pbar = tqdm(total=epochs, desc=f"Rank {rank}", position=rank)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss})
        pbar.update(1)

    pbar.close()

    # 保存模型
    torch.save(state_dict_to_cpu(model.state_dict()), os.path.join(save_path, f"model5_{rank}.pth"))
    return_dict[rank] = state_dict_to_cpu(model.state_dict())  # 返回模型的状态字典并转移到CPU

def create_and_train_model(rank, train_loader, val_loader, epochs, device, return_dict, save_path):
    train_model(rank, train_loader, val_loader, epochs, device, return_dict, save_path)

def main():
    mp.set_start_method('spawn', force=True)  # 使用'spawn'启动方法，并强制设置

    # 初始化共享的字典
    manager = mp.Manager()
    return_dict = manager.dict()

    # 保存路径
    save_path = '/scratch/htc/btian/NNs_test'
    os.makedirs(save_path, exist_ok=True)

    # 集成多个模型
    num_models = 20
    processes = []

    for rank in range(num_models):
        p = mp.Process(target=create_and_train_model, args=(rank, train_loader, val_loader, 5000, 'cuda', return_dict, save_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    models = []
    for i in range(num_models):
        model = SimpleNN()
        model.load_state_dict(return_dict[i])
        model = model.to('cuda')
        models.append(model)

    # 加载测试数据
    Xtest = torch.load('/scratch/htc/btian/NNs_test/test_samples.pt').float().to('cuda')
    Ytest = torch.load('/scratch/htc/btian/NNs_test/ground_truth_test.pt').float().to('cuda')

    # Normalize Xtest using the same fixed min and max values
    Xtest = normalize_tensor(Xtest)

    # 集成预测函数
    def ensemble_predict(models, inputs):
        inputs = inputs.float()
        outputs = [model(inputs) for model in models]
        return torch.mean(torch.stack(outputs), dim=0)

    # 评估集成模型
    ensemble_predictions = ensemble_predict(models, Xtest)
    test_loss = nn.MSELoss()(ensemble_predictions, Ytest)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
