#! -*- coding: utf-8 -*-

from tqdm import tqdm
import os, re, json, codecs
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import tensorflow as tf

mode = 0
maxlen = 128
learning_rate = 5e-5
min_learning_rate = 1e-5

# pretrain_model = '/Users/li/workshop/MyRepository/DeepQ/preTrainedModel/tensorlfow/'
# pretrain_model_name = 'chinese_L-12_H-768_A-12'

pretrain_model = '/home/dqnlp/virtualenv/preTrainedModel/'
pretrain_model_name = 'chinese_wwm_L-12_H-768_A-12'

config_path = pretrain_model + pretrain_model_name + '/bert_config.json'
checkpoint_path = pretrain_model + pretrain_model_name + '/bert_model.ckpt'
dict_path = pretrain_model + pretrain_model_name + '/vocab.txt'

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
D = D[D[2] != u'nan']
classes = list(set(D[2].unique()))

# 将分类目录固定，转换为{类别: id}表示;
categories = set(classes)
categories = [str(x) for x in categories]
cat_to_id = dict(zip(categories, range(len(categories))))
id_to_cat = dict(zip(range(len(categories)), categories))

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


def seq_padding(inputs, length=None, padding=0):
    if length is None:
        length = max([len(x) for x in inputs])
    outputs = np.array(
        [np.concatenate([x, [padding] * (length - len(x))]) if len(x) < length else x[:length] for x in inputs])

    return outputs


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
            batch_token_ids, batch_segment_ids, S1, S2, batch_labels = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, category, label = d[0], d[1], d[2]
                text = text[:maxlen]
                tokens = tokenizer.tokenize(text)
                e_tokens = tokenizer.tokenize(label)[1:-1]
                # 构造输出
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)

                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    token_ids, segment_ids = tokenizer.encode(first=text)
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    S1.append(s1)
                    S2.append(s2)
                    batch_labels.append([cat_to_id[category]])

                    if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                        batch_token_ids = seq_padding(batch_token_ids)
                        batch_segment_ids = seq_padding(batch_segment_ids)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        batch_labels = seq_padding(batch_labels)
                        yield [batch_token_ids, batch_segment_ids, S1, S2, batch_labels], None
                        batch_token_ids, batch_segment_ids, S1, S2, batch_labels = [], [], [], [], []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = tf.gather(output, subject_ids[:, :1], batch_dims=-1)
    end = tf.gather(output, subject_ids[:, 1:], batch_dims=-1)
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 加载预训练模型
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True
bert_model.summary()

q_x_in = Input(shape=(None,), name='Input-Token-query')  # 待识别句子输入
q_s_in = Input(shape=(None,), name='Input-Segment-query')  # 待识别句子输入
q_st_in = Input(shape=(None,), name='Input-Start-query')  # 实体左边界（标签）
q_en_in = Input(shape=(None,), name='Input-End-query')  # 实体右边界（标签）
q_label_in = Input(shape=(None,), name='Input-label-query')  # 实体右边界（标签）

tokens, segments, q_st, q_en, q_label = q_x_in, q_s_in, q_st_in, q_en_in, q_label_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x_mask')(tokens)
bert_output = bert_model([tokens, segments])

# 预测category
x = Lambda(lambda x: x[:, 0], name='CLS_Token')(bert_output)
out1 = Dropout(0.5, name='out1')(x)
ps0 = Dense(units=len(classes), activation='sigmoid', name='ps_category')(out1)

# 利用ps0的信息
# output = bert_output.layers[-2].get_output_at(-1)


ps1 = Dense(1, use_bias=False, name='dps1')(bert_output)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps_heads')([ps1, x_mask])
ps2 = Dense(1, use_bias=False, name='dps2')(bert_output)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps_tails')([ps2, x_mask])
subject_model = Model([q_x_in, q_s_in], [ps0, ps1, ps2])

train_model = Model([q_x_in, q_s_in, q_st_in, q_en_in, q_label_in], [ps0, ps1, ps2])
train_model.summary()

loss0 = K.mean(K.sparse_categorical_crossentropy(q_label, ps0, from_logits=True))
loss1 = K.mean(K.categorical_crossentropy(q_st, ps1, from_logits=True))
ps2 -= (1 - K.cumsum(q_st, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(q_en, ps2, from_logits=True))

loss = loss0 + loss1 + loss2
train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))

if not os.path.exists('../images/model_temp.png'):
    from keras.utils.vis_utils import plot_model

    plot_model(train_model, to_file="../images/model_temp.png", show_shapes=True)


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in):
    # text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps0, _ps1, _ps2 = subject_model.predict([_x1, _x2])
    # print('_ps0: {}'.format(_ps0))
    # print('_ps1: {}'.format(_ps1))
    # print('_ps2: {}'.format(_ps2))
    _ps0, _ps1, _ps2 = softmax(_ps0[0]), softmax(_ps1[0]), softmax(_ps2[0])
    # print('_ps0: {}'.format(_ps0))
    # print('_ps1: {}'.format(_ps1))
    # print('_ps2: {}'.format(_ps2))

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
    return a, id_to_cat[np.argmax(_ps0)]


class Evaluate(Callback):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.ACC = []
        self.best = 0.
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """
        第一个epoch用来warmup，第二个epoch把学习率降到最低
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
            train_model.save_weights('../model/best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R, obj = extract_entity(d[0])
            print('================================')
            print('category_real: {}'.format(d[1]))
            print('category_pre: {}'.format(obj))
            print('============')
            print('object_real: {}'.format(d[2]))
            print('object_pre: {}'.format(R))

            if R == d[2] and obj == d[1]:
                A += 1
            # s = ', '.join(d + (R,))
            # F.write(s.encode('utf-8') + '\n')
        # F.close()
        return A / len(dev_data)
        # return 0


def test():
    D = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', encoding='utf-8', header=None)
    test_data = D[0].apply(lambda x: x.split('\t')).values
    F = open('result.txt', 'w')
    for d in tqdm(iter(test_data)):
        _object, _category = extract_entity(d[1])
        s = '%s\t%s\t%s\n' % (d[0], _category, _object)
        s = s.encode('utf-8')
        F.write(str(s))
    F.close()

evaluator = Evaluate()
train_D = data_generator(train_data)


if __name__ == '__main__':
    if not os.path.exists('../model/best_model.weights'):
        print('Training......')
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[evaluator])
    else:
        print('Testing.......')
        train_model.load_weights('../model/best_model.weights')
        test()