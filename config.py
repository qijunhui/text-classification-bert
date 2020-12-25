# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/22 20:14
@Author  : qijunhui
@File    : config.py
"""
import os

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")  # 数据路径
MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model")  # 模型路径
BERT_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model", "bert_model")  # bert模型路径
TRAIN_DATA_RATIO = 0.9  # 训练数据比例
TEST_DATA_RATIO = 0.05  # 测试数据比例
VAL_DATA_RATIO = 1 - TRAIN_DATA_RATIO - TEST_DATA_RATIO  # 验证数据比例


# 若没有模型路径则自动创建
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
    os.mkdir(BERT_MODEL_PATH)
