# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/26 15:26
@Author  : qijunhui
@File    : model.py
"""
import os
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from config import BERT_MODEL_PATH


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if (
            os.path.exists(os.path.join(BERT_MODEL_PATH, "config.json"))
            and os.path.exists(os.path.join(BERT_MODEL_PATH, "vocab.txt"))
            and os.path.exists(os.path.join(BERT_MODEL_PATH, "pytorch_model.bin"))
        ):
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=BERT_MODEL_PATH)
            self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=BERT_MODEL_PATH)
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences, device="cpu"):
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
