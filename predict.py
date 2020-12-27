# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/27 19:57
@Author  : qijunhui
@File    : predict.py
"""
import os
import torch
from config import MODEL_PATH
from model import Net

net = Net()
net.load_state_dict(torch.load(os.path.join(MODEL_PATH, "net.pkl"), map_location=torch.device("cpu")))
net.eval()


def predict(text):
    outputs = net([text])
    return torch.argmax(outputs, dim=1).item()


if __name__ == "__main__":
    predict("strange , funny , twisted , brilliant and macabre")
