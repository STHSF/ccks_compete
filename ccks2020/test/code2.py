#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: code2.py
@time: 2020/4/23 3:45 下午
"""

import os
import re
import gc
import sys
import json
import codecs
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from random import choice
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report
from collections import Counter
import jieba.analyse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import time
import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras.optimizers import Adam
import operator
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tqdm.pandas()
np.random.seed(214683)
warnings.filterwarnings('ignore')


# In[2]:


def int_entity(x):
    try:
        x = int(x)
    except:
        x
    return x


# In[3]:


def top_2_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


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
    s = re.sub("\u3000", "", s)
    s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/]\d{2}[-/]\d{2}')  # 日期
    s = re.sub(r4, '某时', s)
    return s


def get_data():
    train = pd.read_csv('../data/event_entity_train_data_label.csv', sep='\t', header=None,
                        names=['uid', 'content', 'content_type', 'entity'])
    test = pd.read_csv('../data/event_entity_dev_data.csv', sep='\t', header=None, names=['uid', 'content'])
    task1 = pd.read_csv('../submit/entity_7274_969_666.csv')
    task1 = task1[~task1['entity'].isnull()]
    task1 = task1[task1['entity'].apply(len) < 19].reset_index(drop=True)
    test = task1.merge(test, on='uid', how='left').reset_index(drop=True)
    train = train[~train.content_type.isnull()].drop_duplicates().reset_index(drop=True)
    train['content'] = train['content'].apply(lambda x: delete_tag(x))
    test['content'] = test['content'].apply(lambda x: delete_tag(x))
    train['content'] = list(map(lambda x, y: x[x.find(y) - 150:x.find(y) + 250], train['content'], train['entity']))
    test['content'] = list(map(lambda x, y: x[x.find(y) - 150:x.find(y) + 250], test['content'], test['entity']))
    return train, test


def get_length(x):
    try:
        return len(x)
    except:
        return np.nan


# In[4]:


# path_drive = '../../../weight/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
# config_path = path_drive + 'bert_config.json'
# checkpoint_path = path_drive + 'bert_model.ckpt'
# dict_path = path_drive + 'vocab.txt'

pretrain_model = '/home/dqnlp/virtualenv/preTrainedModel/'
# pretrain_model_name = 'chinese_wwm_L-12_H-768_A-12'
pretrain_model_name = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'

config_path = pretrain_model + pretrain_model_name + '/bert_config.json'
checkpoint_path = pretrain_model + pretrain_model_name + '/bert_model.ckpt'
dict_path = pretrain_model + pretrain_model_name + '/vocab.txt'

# In[5]:


gc.collect()
MAXLEN = 400  # 510 = 5

n_folds = 10
folds_num = str(n_folds) + 'folds_'

strategy = '_withprocess_chusai&fusaidata_'
model = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
date_str = 'chinese_wwm_ext_L-12_H-768_A-12_' + 'maxlen' + str(MAXLEN)
learning_rate = 3e-5
min_learning_rate = 5e-6

# In[6]:


# 给每个token按序编号，构建词表
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 分词器
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


# Padding，默认添 0
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    def __init__(self, data, feature, batch_size=24, shuffle=True):  # 8
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature = feature
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y, Fea = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:MAXLEN]
                fea = self.feature[i]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Fea.append(fea)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Fea = seq_padding(Fea)
                    Y = seq_padding(Y)
                    yield [X1, X2, Fea], Y
                    X1, X2, Y, Fea = [], [], [], []


# In[7]:


# 计算：F1值
def f1_metric(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# In[8]:


# BERT模型建立
def build_bert(nclass, maxlen, train_flag=2):
    #     train_flag == 1
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=maxlen)

    for layer in bert_model.layers:
        #         print(l)
        layer.trainable = True

    # inputs
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(train_feature.shape[1],))

    feature = Dense(32, activation='relu')(x3_in)
    x = bert_model([x1_in, x2_in])
    #     print('Bert output shape', x.shape)
    x = Lambda(lambda x: x[:, 0])(x)
    x = concatenate([x, feature])
    # outputs
    p = Dense(nclass, activation='softmax')(x)

    # 模型建立与编译
    #     model = Model([x1_in, x2_in], p)
    model = Model([x1_in, x2_in, x3_in], p)
    if train_flag == 1:
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy', f1_metric, categorical_accuracy])

    else:
        model = multi_gpu_model(model, gpus=2)  # 使用几张显卡n等于几
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy', f1_metric, categorical_accuracy])
    #     print(model.summary())

    return model


from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score


# In[9]:


class Evaluate(Callback):
    def __init__(self):
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


# In[10]:


def multi_process(func_name, process_num, deal_list):
    """
    多线程
    """
    from multiprocessing import Pool
    pool = Pool(process_num)
    result_list = pool.map(func_name, deal_list)
    pool.close()
    pool.join()
    return result_list


# In[11]:


traindata, testdata = get_data()
traindata['corpus'] = traindata['content']
traindata['label'] = traindata['content_type']
testdata['corpus'] = testdata['content']

traindata['corpus'] = traindata['corpus'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)
testdata['corpus'] = testdata['corpus'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x).fillna('666')

# In[12]:


traindata['label'].unique()


# In[13]:


def get_textrank(x):
    try:
        return jieba.analyse.textrank(x)
    except:
        return np.nan


traindata['text_rank_list'] = multi_process(get_textrank, 40, traindata['corpus'])
testdata['text_rank_list'] = multi_process(get_textrank, 40, testdata['corpus'])
# 寻找词库
all_word = list()
for i in traindata['text_rank_list']:
    try:
        all_word.extend(i)
    except:
        continue
for i in testdata['text_rank_list']:
    try:
        all_word.extend(i)
    except:
        continue
# 统计词频
dic = {}
for word in all_word:
    if word not in dic:
        dic[word] = 1
    else:
        dic[word] = dic[word] + 1
swd = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)


# In[14]:


def get_frequency(x):
    #     try:
    fre = list()
    for i in x:
        num = dic[i]
        fre.append(num)
    return fre


def get_max(x):
    try:
        return max(x)
    except:
        return np.nan


def get_min(x):
    try:
        return min(x)
    except:
        return np.nan


def get_mean(x):
    try:
        return np.mean(x)
    except:
        return np.nan


def get_std(x):
    try:
        return np.std(x)
    except:
        return np.nan


def get_ptp(x):
    try:
        return np.ptp(x)
    except:
        return np.nan


def get_sum(x):
    try:
        return np.sum(x)
    except:
        return np.nan


# In[15]:


traindata['text_rank_list_fre'] = traindata['text_rank_list'].apply(lambda x: get_frequency(x))
testdata['text_rank_list_fre'] = testdata['text_rank_list'].apply(lambda x: get_frequency(x))

# In[16]:


fun_dict = {'max': get_max, 'mean': get_mean, 'min': get_min, 'sum': get_sum, 'ptp': get_ptp, 'std': get_std}
for i in ['max', 'mean', 'min', 'sum', 'ptp', 'std']:
    traindata['text_rank_list_' + i] = multi_process(fun_dict[i], 40, traindata['text_rank_list_fre'])
    testdata['text_rank_list_' + i] = multi_process(fun_dict[i], 40, testdata['text_rank_list_fre'])
traindata['corpus_len'] = traindata['corpus'].apply(lambda x: len(x))
testdata['corpus_len'] = testdata['corpus'].apply(lambda x: len(x))

# In[17]:


fea_col = [i for i in traindata.columns if 'text_rank_list_' in i]
fea_col.append('corpus_len')

# In[18]:


train_feature = traindata[fea_col[1:]]
test_feature = testdata[fea_col[1:]]

# In[19]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_feature = scaler.fit_transform(train_feature.fillna(-1))
test_feature = scaler.fit_transform(test_feature.fillna(-1))

# In[20]:


traindata['corpus'] = traindata['entity'] + traindata['text_rank_list'].apply(lambda x: ','.join(x)) + traindata[
    'corpus'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)
testdata['corpus'] = testdata['entity'] + testdata['text_rank_list'].apply(lambda x: ','.join(x)) + testdata[
    'corpus'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x).fillna('666')

# In[21]:


label_map = {j: i for i, j in enumerate(traindata.label.unique())}
label_map_iv = {i: j for i, j in enumerate(traindata.label.unique())}
labelcol = list(traindata.label.unique())

# In[22]:


traindata['label'] = traindata['label'].map(label_map)

# In[23]:


DATA_LIST = []
for data_row in traindata.iloc[:].itertuples():
    DATA_LIST.append([data_row.corpus, to_categorical(data_row.label, 27)])
DATA_LIST = pd.DataFrame(DATA_LIST).values

DATA_LIST_TEST = []
for data_row in testdata.iloc[:].itertuples():
    DATA_LIST_TEST.append([data_row.corpus, to_categorical(0, 27)])
DATA_LIST_TEST = pd.DataFrame(DATA_LIST_TEST).values


# In[24]:


def run_cv(nfolds, feature_train, data, data_label, data_test, feature_test, epochs=10, date_str='1107', m=167,
           n_class=27, col_y=labelcol):
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=2114683).split(data, traindata['label'])
    train_model_pred = np.zeros((len(data), n_class))
    test_model_pred = np.zeros((len(data_test), n_class))

    for i, (train_fold, test_fold) in enumerate(skf):
        print('Fold: ', i + 1)

        '''数据部分'''
        # 数据划分
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        y_val = traindata['label'][test_fold]
        X_train_fea, X_valid_fea = feature_train[train_fold, :], feature_train[test_fold, :]

        train_D = data_generator(X_train, X_train_fea, shuffle=True)
        valid_D = data_generator(X_valid, X_valid_fea, shuffle=False)
        test_D = data_generator(data_test, feature_test, shuffle=False)

        time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        '''模型部分'''
        # 生成模型
        model = build_bert(nclass=n_class, maxlen=m + 2)
        # callbacks
        early_stopping = EarlyStopping(monitor='val_f1_metric', patience=3)  # val_acc
        plateau = ReduceLROnPlateau(monitor="val_f1_metric", verbose=1, mode='max', factor=0.5,
                                    patience=1)  # max：未上升则降速
        checkpoint = ModelCheckpoint('../model/' + date_str + str(i) + '.hdf5', monitor='val_f1_metric',
                                     verbose=2, save_best_only=True, mode='max',
                                     save_weights_only=True)  # period=1: 每1轮保存

        evaluator = Evaluate()
        file_path = '../model/' + date_str + str(i) + '.hdf5'
        #         模型训练，使用生成器方式训练
        if not os.path.exists(file_path):
            model.fit_generator(
                train_D.__iter__(),
                steps_per_epoch=len(train_D),  ## ?? ##
                epochs=epochs,
                validation_data=valid_D.__iter__(),
                validation_steps=len(valid_D),
                callbacks=[early_stopping, plateau, checkpoint, evaluator],  # evaluator,
                verbose=1
            )
            model.load_weights(file_path)
        else:
            model.load_weights(file_path)

        # return model
        val = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=0)
        pred_val = np.argmax(val, axis=1)

        print(classification_report(y_val, pred_val, target_names=col_y))

        train_model_pred[test_fold, :] = val
        pred = model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
        test_model_pred = pred + test_model_pred
        #         print(pred)

        del model
        gc.collect()

        K.clear_session()
        # break

    return train_model_pred, test_model_pred


# In[25]:


start_time = time.time()

train_model_pred, test_model_pred = run_cv(n_folds, train_feature, DATA_LIST, None, DATA_LIST_TEST, test_feature,
                                           epochs=8, date_str=date_str, m=MAXLEN)
# print('Validate 5folds average f1 score:', np.average(f1))


# In[26]:


np.save('../submit/tr' + model + strategy + folds_num + date_str + '.npy', train_model_pred)
np.save('../submit/te' + model + strategy + folds_num + date_str + '.npy', test_model_pred)

end_time = time.time()
print('Time cost(min): ', (end_time - start_time) / 60)
submit = testdata[['uid', 'entity']].copy()
submit['label'] = np.argmax(test_model_pred, axis=1)
# submit['label'] = submit['label'].replace(2,-1)
submit['label'] = submit['label'].map(label_map_iv)
submit = submit[['uid', 'label', 'entity']].copy()
submit.to_csv('../submit/' + model + strategy + folds_num + date_str + '.csv', index=False, header=None)

# In[29]:


sub_example = pd.read_csv('../submit/result_duo_416_726.csv', sep='\t', header=None, names=['uid', 'label', 'entity'])
sub = sub_example[['uid']].merge(submit, on='uid', how='left')
sub = sub.drop_duplicates().reset_index(drop=True)
sub.to_csv('../submit/' + model + '_420_v2_' + folds_num + '.csv', header=None, index=None, sep='\t')