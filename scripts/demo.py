# -*- coding: utf-8 -*-
"""
@Time    : 2021/1/13 18:58
@Author  : qijunhui
@File    : demo.py
"""
# 参考：https://www.cnblogs.com/cxq1126/p/13517394.html
import os
import torch
from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
)

model_name = "bert-base-chinese"
bert_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model", "chinese_bert_model")


def task_1():
    # 任务一：遮蔽语言模型
    sample = ("中国的首都是哪里？", "北京是[MASK]国的首都。")  # 准备输入模型的语句 [MASK]必须是大写

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sen_code = tokenizer.batch_encode_plus([sample])  # 上下句结合可以这样传参 List[Tuple[str, str]]
    input_ids = torch.tensor(sen_code["input_ids"])

    model = BertForMaskedLM.from_pretrained(bert_path)

    model.eval()
    outputs = model(input_ids)
    prediction_scores = outputs.logits  # torch.Size([batch, max_length, vocab_num])

    # pred_lst = prediction_scores.max(dim=2).indices  # torch.Size([batch, max_length])
    pred_lst = prediction_scores.argmax(axis=2)  # torch.Size([batch, max_length])
    for pred in pred_lst:
        print(f"b被屏蔽掉的字是：{tokenizer.convert_ids_to_tokens(pred)[14]}")  # 被标记的[MASK]是第14个位置


def task_2():
    # 任务二：句子预测任务
    sample_1 = ("今天天气怎么样", "今天天气很好")
    sample_2 = ("小明今年几岁了", "我不喜欢学习")

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sen_code = tokenizer.batch_encode_plus([sample_1, sample_2])  # 上下句结合可以这样传参 List[Tuple[str, str]]
    input_ids = torch.tensor(sen_code["input_ids"])

    model = BertForNextSentencePrediction.from_pretrained(bert_path)

    model.eval()
    outputs = model(input_ids)
    seq_relationship_scores = outputs.logits  # torch.Size([batch, 2])

    # pred_lst = seq_relationship_scores.max(dim=1).indices  # torch.Size([batch, 2])
    pred_lst = seq_relationship_scores.argmax(axis=1)  # torch.Size([batch, 2])
    for pred in pred_lst:
        print(f"预测结果：{pred}")  # 0表示是上下句，1表示不是上下句（第二句明明不是前后句关系，不知道为什么会输出0）


def task_3():
    # 任务三：句子预测任务
    question, text = "里昂是谁", "里昂是一个杀手"
    sample = (question, text)

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sen_code = tokenizer.batch_encode_plus([sample])  # 上下句结合可以这样传参 List[Tuple[str, str]]
    tokens_tensor = torch.tensor(sen_code["input_ids"])
    segments_tensor = torch.tensor(sen_code["token_type_ids"])

    model_config = BertConfig.from_pretrained(bert_path)
    # model_config.num_labels = 2  # 最终有两个输出，初始位置和结束位置
    # model = BertForQuestionAnswering.from_pretrained(bert_path)  # 这是一种加载方式
    model = BertForQuestionAnswering(model_config)  # 这是另一种加载方式

    model.eval()
    outputs = model(tokens_tensor, segments_tensor)
    start_pos, end_pos = outputs.start_logits, outputs.end_logits

    for idx, (start, end) in enumerate(zip(start_pos.argmax(axis=1), end_pos.argmax(axis=1))):
        all_tokens = tokenizer.convert_ids_to_tokens(sen_code["input_ids"][idx])  # 进行逆编码，得到原始的token
        print(all_tokens)  # ['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']
        if start <= end:
            answer = " ".join(all_tokens[start : end + 1])  # 对输出的答案进行解码的过程
            # 每次执行的结果不一致，这里因为没有经过微调，所以效果不是很好，输出结果不佳，下面的输出是其中的一种。
            print(answer)  # 一 个 杀 手 [SEP]
        else:
            print("预测的有问题哦！")


if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
