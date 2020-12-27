# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/22 20:14
@Author  : qijunhui
@File    : train.py
"""
# 参考：https://blog.csdn.net/a553181867/article/details/105389757/
import os
import torch
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score
from math import ceil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import DATA_PATH, MODEL_PATH
from utils import read_csv
from model import Net


SUMMARY_WRITER = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有gpu则使用gpu

train_datasets = read_csv(os.path.join(DATA_PATH, "train.tsv"), filter_title=True, delimiter="\t")
train_texts = [text for text, target in train_datasets]
train_targets = [int(target) for text, target in train_datasets]

val_datasets = read_csv(os.path.join(DATA_PATH, "val.tsv"), filter_title=True, delimiter="\t")
val_texts = [text for text, target in val_datasets]
val_targets = [int(target) for text, target in val_datasets]

batch_size = 50
epochs = 10
lr = 0.003
print_every_batch = 5
train_batch_count = ceil(len(train_datasets) / batch_size)
val_batch_count = ceil(len(val_datasets) / batch_size)

net = Net().to(device)

##----- 1. 不训练bert层  -----##
for k, v in net.named_parameters():
    if k.startswith("bert"):
        v.requires_grad = False  # 固定参数 不训练bert层
    else:
        pass
optimizer = optim.Adam(net.parameters(), lr=lr)
##----- 直接不训练bert层  -----##

##----- 2. 微调bert层  -----##
# dense_params = list(map(id, net.dense.parameters()))
# base_params = filter(lambda p: id(p) not in dense_params, net.parameters())
# optimizer = optim.Adam([{"params": base_params}, {"params": net.dense.parameters(), "lr": lr * 100}], lr=lr)
##----- 微调bert层  -----##

criterion = nn.CrossEntropyLoss()  # 损失函数

with tqdm(iterable=range(epochs), desc="进度条", ncols=150) as bar:
    for epoch in bar:
        print_avg_loss = 0
        train_acc = 0
        val_acc = 0

        net.eval()  # 预测
        for i in range(val_batch_count):
            inputs = val_texts[i * batch_size : (i + 1) * batch_size]
            labels = torch.tensor(val_targets[i * batch_size : (i + 1) * batch_size]).to(device)
            outputs = net(inputs, device=device)
            val_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), labels.cpu())
        val_acc = val_acc / val_batch_count

        net.train()  # 训练
        for i in range(train_batch_count):
            inputs = train_texts[i * batch_size : (i + 1) * batch_size]
            labels = torch.tensor(train_targets[i * batch_size : (i + 1) * batch_size]).to(device)

            outputs = net(inputs, device=device)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_step = epoch * train_batch_count + i
            if batch_step % print_every_batch == 0 and i != 0:
                bar.set_postfix(
                    {
                        "batch_step": f"{batch_step}",
                        "lr": optimizer.param_groups[0]["lr"],  # 如果为不同层设置不同的学习率，则修改index即可
                        "loss": f"{print_avg_loss / i:.4f}",
                        "train_acc": f"{train_acc / i:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                SUMMARY_WRITER.add_scalar(tag="loss", scalar_value=print_avg_loss / i, global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="train_acc", scalar_value=train_acc / i, global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="val_acc", scalar_value=val_acc, global_step=batch_step)
            print_avg_loss += loss.item()
            train_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), labels.cpu())
