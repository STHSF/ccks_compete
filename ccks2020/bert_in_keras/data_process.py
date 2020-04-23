#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: data_process.py
@time: 2020/4/23 4:33 下午
"""
import re
import pandas as pd
import numpy as np


def delete_tag(s):
    s = re.sub('\{IMG:.?.?.?\}', '', s)  # 图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)  # 网址
    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签
    s = re.sub(re.compile('&[a-zA-Z]+;?'), ' ', s)  # 网页标签
    s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), ' ', s)
    s = re.sub("\?{2,}", "", s)
    s = re.sub("\r", "", s)
    s = re.sub("\n", ",", s)
    s = re.sub("\t", ",", s)
    s = re.sub("（", ",", s)
    s = re.sub("）", ",", s)
    s = re.sub("\u3000", "", s)
    s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/]\d{2}[-/]\d{2}')  # 日期
    s = re.sub(r4, '某时', s)
    return s


def cut_sentences(content):
    # 结束符号，包含中文和英文的
    end_flag = ['。', ';', '；']

    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # 拼接字符
        tmp_char += char

        # 判断是否已经到了最后一位
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break

        # 判断此字符是否为结束符号
        if char in end_flag:
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''

    return sentences


def metl_data(df):
    z = df.groupby(['uid'])['content'].apply(lambda x: np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['uid'], row['content']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['content', 'uid']
    return i


def get_data():
    train = pd.read_csv('../ccks2020Data/event_entity_train_data_label.csv', sep='\t', header=None,
                        names=['uid', 'content', 'content_type', 'entity'])
    test = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', header=None)
    test = test[0].apply(lambda x: x.split('\t')).values.tolist()
    test = pd.DataFrame(test, columns=['uid', 'content'])
    train = train[~train.content_type.isnull()].drop_duplicates().reset_index(drop=True)
    train['content'] = train['content'].apply(lambda x: cut_sentences(str(x)))
    train['content'] = list(map(lambda x, y: [i for i in x if y in i], train['content'], train['entity']))
    train_n = metl_data(train)
    train = train_n.merge(train[['uid', 'entity']], how='left')
    test['content'] = test['content'].apply(lambda x: cut_sentences(str(x)))
    test = metl_data(test)
    train['content'] = train['content'].apply(lambda x: delete_tag(x))
    test['content'] = test['content'].apply(lambda x: delete_tag(x))

    train['content'] = list(map(lambda x, y: x[x.find(y) - 200:x.find(y) + 200], train['content'], train['entity']))
    return train, test