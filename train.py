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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from math import ceil
from sklearn.metrics import accuracy_score
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
    train_acc = 0
    val_acc = 0
    max_val_acc = 0
    batch_step = 0
    for epoch in bar:
        print_avg_loss = 0

        net.train()  # 训练
        train_acc = 0
        for index in range(ceil(len(train_datasets) / batch_size)):
            texts = train_texts[index * batch_size : (index + 1) * batch_size]
            targets = torch.tensor(train_targets[index * batch_size : (index + 1) * batch_size]).to(device)

            outputs = net(texts, device=device)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()
            train_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu())
            batch_step += 1

            if batch_step % print_every_batch == 0:
                bar.set_postfix(
                    {
                        "batch_step": f"{batch_step}",
                        "lr": optimizer.param_groups[0]["lr"],  # 如果为不同层设置不同的学习率，则修改index即可
                        "loss": f"{print_avg_loss / (index + 1):.4f}",
                        "train_acc": f"{train_acc / (index + 1):.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                SUMMARY_WRITER.add_scalar(tag="loss", scalar_value=print_avg_loss / (index + 1), global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="train_acc", scalar_value=train_acc / (index + 1), global_step=batch_step)
                SUMMARY_WRITER.add_scalar(tag="val_acc", scalar_value=val_acc, global_step=batch_step)

        net.eval()  # 预测
        val_acc = 0
        val_batch_count = 0
        for index in range(ceil(len(val_datasets) / batch_size)):
            texts = val_texts[index * batch_size : (index + 1) * batch_size]
            targets = torch.tensor(val_targets[index * batch_size : (index + 1) * batch_size]).to(device)
            outputs = net(texts, device=device)
            val_acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), targets.cpu())
            val_batch_count += 1
        val_acc = val_acc / val_batch_count

        if max_val_acc < val_acc:
            max_val_acc = val_acc
            torch.save(
                net.state_dict(),
                os.path.join(MODEL_PATH, f"{epoch}-train_acc{train_acc:.4f}-val_acc{val_acc:.4f}-net.pkl"),
            )
