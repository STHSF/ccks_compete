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
# print(trainData[2])
# trainData = trainData.dropna()
# classes = set(trainData[2].unique())
# train_data = []
# print(classes)
#
# print(set(trainData[2].values))



D = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', encoding='utf-8', header=None)

test_data = D[0].apply(lambda x: x.split('\t')).values.tolist()
print(pd.DataFrame(test_data, columns=['uid', 'content']))