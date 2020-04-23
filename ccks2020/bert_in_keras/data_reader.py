#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: data_reader.py
@time: 2020/4/20 7:54 下午
"""
import pandas as pd
from tqdm import tqdm

D = pd.read_csv('../ccks2020Data/event_entity_train_data_label.csv', encoding='utf-8', header=None, sep='\t')
# print(D[2])
# D = D.dropna()
# classes = set(D[2].unique())
# train_data = []
# print(classes)
#
# print(set(D[2].values))

D = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', encoding='utf-8', header=None)

# print(D[6].values)
test_data = D[0].apply(lambda x: x.split('\t')).values

# for id, text in zip(id, text):
#     test_data.append((id, text))
# print(test_data)
# F = open('result.txt', 'w')
F = open('result.csv', 'w')
for d in tqdm(iter(test_data)):
    s = '%s\t%s\n' % (d[0], d[1])
    s = s.encode('utf-8')
    print(s)
    F.write(str(s))
F.close()
