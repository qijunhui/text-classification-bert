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
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from math import ceil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import DATA_PATH, BERT_MODEL_PATH
from utils import read_csv


SUMMARY_WRITER = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有gpu则使用gpu


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=BERT_MODEL_PATH)
        # self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=BERT_MODEL_PATH)
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences, device):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences, add_special_tokens=True, padding="max_length", truncation=True, max_length=64
        )  # add_special_tokens 在句首添加[CLS]  padding 填充  truncation 截断  max_length 最大句长
        # print(self.tokenizer.convert_ids_to_tokens(batch_tokenized["input_ids"][0]))  # 可以将编码后的input_ids转化为文本
        # input_ids  torch.Size([batch, max_length])  [[idx, idx, ...],...]  idx为单词index  填充为0
        input_ids = torch.tensor(batch_tokenized["input_ids"]).to(device)
        # input_ids  torch.Size([batch, max_length])  [[1, 1, ...],...]  1的个数为单词的个数  即input_ids中不为0的个数  其余填充为0
        attention_mask = torch.tensor(batch_tokenized["attention_mask"]).to(device)
        # 元组 有两个值  bert_output[0] torch.Size([batch, max_length, 768])  bert_output[1] torch.Size([batch, 768])
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态 torch.Size([batch, 768])
        linear_output = self.dense(bert_cls_hidden_state)  # torch.Size([batch, 2])
        return linear_output


train_datasets = read_csv(os.path.join(DATA_PATH, "train.tsv"), filter_title=True, delimiter="\t")
train_texts = [text for text, target in train_datasets]
train_targets = [int(target) for text, target in train_datasets]


batch_size = 200
batch_count = ceil(len(train_datasets) / batch_size)
epochs = 500
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

with tqdm(iterable=range(epochs), desc="进度条", ncols=100) as bar:
    for epoch in bar:
        net.train()
        print_avg_loss = 0
        acc = 0
        for i in range(batch_count):
            inputs = train_texts[i * batch_size : (i + 1) * batch_size]
            labels = torch.tensor(train_targets[i * batch_size : (i + 1) * batch_size]).to(device)

            outputs = net(inputs, device=device)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch * batch_count + i) % print_every_batch == 0 and i != 0:
                bar.set_postfix(
                    {
                        "batch": f"{epoch * batch_count + i}",
                        "lr": optimizer.param_groups[0]["lr"],  # 如果为不同层设置不同的学习率，则修改index即可
                        "loss": f"{print_avg_loss / i:.4f}",
                        "acc": f"{acc / i:.4f}",
                    }
                )
                SUMMARY_WRITER.add_scalar(
                    tag="loss", scalar_value=print_avg_loss / i, global_step=epoch * batch_count + i
                )
                SUMMARY_WRITER.add_scalar(tag="acc", scalar_value=acc / i, global_step=epoch * batch_count + i)
            print_avg_loss += loss.item()
            acc += accuracy_score(torch.argmax(outputs, dim=1).cpu(), labels.cpu())
