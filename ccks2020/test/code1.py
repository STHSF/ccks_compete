#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: code1.py
@time: 2020/4/23 3:44 下午
"""

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import gc
from keras.layers import *
from keras.models import Model
from keras.utils import multi_gpu_model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


# In[2]:


def top_2_acc(y_true, y_pred):
    return mt.top_k_categorical_accuracy(y_true, y_pred, k=2)


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
    test = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', sep='\t', header=None, names=['uid', 'content'])
    train = train[~train.content_type.isnull()].drop_duplicates().reset_index(drop=True)
    train['content'] = train['content'].apply(lambda x: cut_sentences(x))
    train['content'] = list(map(lambda x, y: [i for i in x if y in i], train['content'], train['entity']))
    train_n = metl_data(train)
    train = train_n.merge(train[['uid', 'entity']], how='left')
    test['content'] = test['content'].apply(lambda x: cut_sentences(x))
    test = metl_data(test)
    train['content'] = train['content'].apply(lambda x: delete_tag(x))
    test['content'] = test['content'].apply(lambda x: delete_tag(x))

    train['content'] = list(map(lambda x, y: x[x.find(y) - 200:x.find(y) + 200], train['content'], train['entity']))
    return train, test


def get_length(x):
    try:
        return len(x)
    except:
        return np.nan


# In[3]:


# train, test = get_data()


# In[4]:


# path_drive = '../../../weight/chinese_wwm_ext_L-12_H-768_A-12/'
# config_path = path_drive + 'bert_config.json'
# checkpoint_path = path_drive + 'bert_model.ckpt'
# dict_path = path_drive + 'vocab.txt'

pretrain_model = '/home/dqnlp/virtualenv/preTrainedModel/'
# pretrain_model_name = 'chinese_wwm_L-12_H-768_A-12'
pretrain_model_name = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'

config_path = pretrain_model + pretrain_model_name + '/bert_config.json'
checkpoint_path = pretrain_model + pretrain_model_name + '/bert_model.ckpt'
dict_path = pretrain_model + pretrain_model_name + '/vocab.txt'


mode = 0
maxlen = 400
learning_rate = 1e-5
min_learning_rate = 1e-6

gc.collect()
# MAXLEN = 172 # 510 = 5
mode = 0
n_folds = 8
folds_num = str(n_folds) + 'folds_'
date_str = '325_ep_3'
strategy = '_withprocess_chusai&fusaidata_'
model = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
date_str = model + '_412'

# In[5]:


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


def bert_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）

    x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

    x = bert_model([x1, x2])
    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    model = Model([x1_in, x2_in], [ps1, ps2])
    model = multi_gpu_model(model, gpus=2)

    train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
    ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
    loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    train_model = multi_gpu_model(train_model, gpus=2)
    train_model.compile(optimizer=Adam(learning_rate))
    #     train_model.summary()
    return model, train_model


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


# In[6]:


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# In[7]:


class data_generator:
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = np.arange(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                #                 text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[1]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []


# In[8]:


def extract_entity(text_in):
    text_in = text_in[:maxlen]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2 = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end + 1].argmax() + start
    a = text_in[start - 1: end]
    return a


def extract_entity_test(text_in):
    #     text_in = u'___%s___%s' % (c_in, text_in)
    # text_in = text_in[:maxlen]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2 = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    tg_list = list()
    for i in range(1):
        if i > 0:
            if sorted(_ps1, reverse=True)[i] > 0.45:
                start = np.argwhere((_ps1 == sorted(_ps1, reverse=True)[i]))[0][0]
                for end in range(start, len(_tokens)):
                    _t = _tokens[end]
                    if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
                        break
                end = _ps2[start:end + 1].argmax() + start
                a = text_in[start - 1: end]
                #                 if i>0 and len(a)<10:
                tg_list.append(a)
                tg_list.append(a)
        else:
            start = np.argwhere((_ps1 == sorted(_ps1, reverse=True)[i]))[0][0]
            for end in range(start, len(_tokens)):
                _t = _tokens[end]
                if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
                    break
            end = _ps2[start:end + 1].argmax() + start
            a = text_in[start - 1: end]
            tg_list.append(a)
            tg_list.append(a)
        tg_list = list(set(tg_list))
    return tg_list


# In[9]:


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0])
            if R == d[1]:
                A += 1
            s = ', '.join(d + (R,))
            F.write(s + '\n')
        F.close()
        return A / len(dev_data)


def test(test_data):
    sub = list()
    for d in tqdm(iter(test_data)):
        sub.append(extract_entity_test(d[1]))
    testData['entity'] = sub
    return testData


# In[18]:


trainData, testData = get_data()

# In[21]:


trainData, testData = get_data()
D = trainData
train_data = []
for c, e in zip(D['content'], D['entity']):
    train_data.append((c, e))

random_order = np.arange(len(train_data))
dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 11 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 11 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[0]))

D = testData
test_data = []
for t, c in zip(D['uid'], D['content']):
    test_data.append((t, c))

# In[22]:


evaluator = Evaluate()
train_D = data_generator(train_data)

# In[23]:


model, train_model = bert_model()

# In[ ]:


train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=2,
                          callbacks=[evaluator]
                          )

# In[57]:


sub = test(test_data)


# In[58]:


def recommendation_user_list():
    z = sub.groupby(['uid'])['entity'].apply(lambda x: np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['uid'], row['entity']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['entity', 'uid']
    return i


# In[59]:


submit = recommendation_user_list()

# In[66]:


submit.to_csv('../submit/entity_7274_969_666.csv', index=False)