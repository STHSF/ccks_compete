#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: gene.py
@time: 2020/4/20 7:35 下午
"""
import numpy as np

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs


mode = 0
maxlen = 128
learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

D = pd.read_csv('../ccks2020Data/event_entity_train_data_label.csv', encoding='utf-8', header=None, sep='\t')
D = D[D[2] != u'其他']
classes = set(D[2].unique())
train_data = []
for t, c, n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))

if not os.path.exists('../random_order_train.json'):
    random_order = list(range(len(train_data)))
    print('random_order {}'.format(type(random_order)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_train.json'))

dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))
additional_chars.remove(u'，')


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, S1, S2 = [], [], [], []
            batch_token_ids, batch_segment_ids, S1, S2, batch_labels = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, category, label = d[0], d[1], d[2]
                print(category, text, label)
                text = text[:maxlen]
                print('text {}'.format(text))
                tokens = tokenizer.tokenize(text)
                e = label
                print('label{}'.format(e))
                e_tokens = tokenizer.tokenize(e)[1:-1]
                # 构造输出
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                print('start {}'.format(start))

                if start != -1:
                    end = start + len(e_tokens) - 1
                    print('end {}'.format(end))
                    s1[start] = 1
                    s2[end] = 1
                    token_ids, segment_ids = tokenizer.encode(first=text)
                    print(s1)
                    print(s2)
                    print(token_ids)
                    print(segment_ids)

                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    S1.append(s1)
                    S2.append(s2)
                    if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                        batch_token_ids = seq_padding(batch_token_ids)
                        batch_segment_ids = seq_padding(batch_segment_ids)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [batch_token_ids, batch_segment_ids, S1, S2], batch_labels
                        batch_token_ids, batch_segment_ids, S1, S2, batch_labels = [], [], [], [], []


train_D = data_generator(train_data)

for i in train_D.__iter__():
    i