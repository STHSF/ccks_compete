#! -*- coding: utf-8 -*-

import json, os, re, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.layers import Loss, Dropout, Input, Dense, Lambda, Reshape
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from keras.callbacks import Callback
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, Model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm


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

# 建立分词器
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


tokenizer = OurTokenizer(dict_path, do_lower_case=True)

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

D = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', encoding='utf-8', header=None)
test_data = []
for id, t, c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


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
            batch_token_ids, batch_segment_ids, subject_start, subject_end, batch_labels = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, category, label = d[0], d[1], d[2]
                text = text[:maxlen]
                tokens = tokenizer.tokenize(text)
                e_tokens = tokenizer.tokenize(label)[1:-1]
                # 构造输出
                sub_start, sub_end = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)

                if start != -1:
                    end = start + len(e_tokens) - 1
                    sub_start[start] = 1
                    sub_end[end] = 1
                    token_ids, segment_ids = tokenizer.encode(first_text=text)
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    subject_start.append(sub_start)
                    subject_end.append(sub_end)
                    batch_labels.append([cat_to_id[category]])

                    if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)
                        subject_start = sequence_padding(subject_start)
                        subject_end = sequence_padding(subject_end)
                        batch_labels = sequence_padding(batch_labels)
                        yield [batch_token_ids, batch_segment_ids, subject_start, subject_end, batch_labels], None
                        batch_token_ids, batch_segment_ids, subject_start, subject_end, batch_labels = [], [], [], [], []


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
bert_model = build_transformer_model(config_path=config_path,
                                     checkpoint_path=checkpoint_path,
                                     return_keras_model=False,)

q_x_in = Input(shape=(None,), name='Input-Token-query')  # 待识别句子输入
q_s_in = Input(shape=(None,), name='Input-Segment-query')  # 待识别句子输入
q_start_in = Input(shape=(None,), name='Input-Start-query')  # 实体左边界（标签）
q_end_in = Input(shape=(None,), name='Input-End-query')  # 实体右边界（标签）
q_label_in = Input(shape=(None,), name='Input-label-query')  # 实体右边界（标签）

# tokens, segments, q_st, q_en, q_label = q_x_in, q_s_in, q_start_in, q_end_in, q_label_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='x_mask')(bert_model.model.inputs[0])

# 预测category
x = Lambda(lambda x: x[:, 0], name='CLS_Token')(bert_model.model.output)
out1 = Dropout(0.5, name='out1')(x)
ps_category = Dense(units=len(classes), activation='sigmoid', name='ps_category')(out1)

# 利用ps_category的信息
output = bert_model.model.layers[-2].get_output_at(-1)

ps_heads = Dense(1, activation='sigmoid', use_bias=False, name='dps1')(bert_model.model.output)
ps_heads = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps_heads')([ps_heads, x_mask])
ps_tails = Dense(1, activation='sigmoid', use_bias=False, name='dps2')(bert_model.model.output)
ps_tails = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='ps_tails')([ps_tails, x_mask])

subject_model = Model(bert_model.model.inputs, [ps_category, ps_heads, ps_tails])


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """

    def compute_loss(self, inputs, mask=None):
        q_start_in, q_end_in, q_label_in, ps_category, ps_heads, ps_tails = inputs
        if mask is None:
            mask = 1.0
        else:
            mask = K.cast(mask, K.floatx())
        loss0 = K.sparse_categorical_crossentropy(q_label_in, ps_category, from_logits=True)
        loss0 = K.mean(loss0)
        loss0 = K.sum(loss0 * mask) / K.sum(mask)


        loss1 = K.categorical_crossentropy(q_start_in, ps_heads, from_logits=True)
        loss1 = K.mean(loss1)

        ps_tails = ps_tails - (1 - K.cumsum(q_start_in, axis=1)) * 1e10
        loss2 = K.mean(K.categorical_crossentropy(q_end_in, ps_tails, from_logits=True))
        loss2 = K.mean(loss2)
        # 总的loss
        return loss0 + loss1 + loss2

ps_category, ps_heads, ps_tails = TotalLoss([3,4,5])([q_start_in, q_end_in, q_label_in, ps_category, ps_heads, ps_tails])
train_model = Model(bert_model.model.inputs + [q_start_in, q_end_in, q_label_in], [ps_category, ps_heads, ps_tails])
train_model.summary()

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=learning_rate)
train_model.compile(optimizer=optimizer)

if not os.path.exists('../images/model_temp_b4k.png'):
    from keras.utils.vis_utils import plot_model
    plot_model(train_model, to_file="../images/model_temp_b4k.png", show_shapes=True)


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in):
    # text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first_text=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps0, _ps1, _ps2 = subject_model.predict([_x1, _x2])
    _ps0, _ps1, _ps2 = softmax(_ps0[0]), softmax(_ps1[0]), softmax(_ps2[0])

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
            train_model.save_weights('../model/best_model_b4k.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        B = 1e-10
        C = 1e-10
        for doc in tqdm(iter(dev_data)):
            _object, _category = extract_entity(doc[0])
            # print('================================')
            # print('category_real: {}'.format(d[1]))
            # print('object_real: {}'.format(d[2]))
            # print('============')
            # print('category_pre: {}'.format(obj))
            # print('object_pre: {}'.format(R))

            if _object == doc[2]:
                # 事件主体
                C += 1
            if _category == doc[1]:
                # 事件类型
                B += 1
            if _object == doc[2] and _category == doc[1]:
                A += 1
        category_acc = B / len(dev_data)
        object_acc = C / len(dev_data)
        total_acc = A / len(dev_data)
        return total_acc, object_acc, category_acc


def test(test_data=None):
    if test_data is None:
        test_data = pd.read_csv('../ccks2020Data/event_entity_dev_data.csv', encoding='utf-8', header=None)
    else:
        test_data = test_data
    test_data = test_data[0].apply(lambda x: x.split('\t')).values
    result = []
    for doc in tqdm(iter(test_data)):
        _object, _category = extract_entity(doc[1])
        result.append([doc[0], _category, _object])
    result_df = pd.DataFrame(result)
    result_df.to_csv('result.csv', encoding='utf-8', sep='\t', index=False, header=False)


evaluator = Evaluate()
train_D = data_generator(train_data)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', dest='is_train',
                        default=True,
                        type=bool, help="train or test")
    args = parser.parse_args()
    is_train = args.is_train

    if is_train:
        print('Training......')
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[evaluator])
    else:
        print('Testing.......')
        train_model.load_weights('../model/best_model.weights')
        test()