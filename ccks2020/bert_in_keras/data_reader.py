#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: data_reader.py
@time: 2020/4/20 7:54 下午
"""
import pandas as pd

D = pd.read_csv('../ccks2020Data/event_entity_train_data_label.csv', encoding='utf-8', header=None, sep='\t')
print(D[2])
D = D.dropna()
classes = set(D[2].unique())
train_data = []
print(classes)

print(set(D[2].values))