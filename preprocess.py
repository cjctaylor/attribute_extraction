#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import train_test_split


# embedding the position
def pos_embed(x):
    if x < -70:
        return 0
    if -70 <= x <= 70:
        return x + 71
    if x > 70:
        return 142


def data2array(file, word2id: dict, relation2id: dict):
    train_x = []
    train_y = []
    fix_sentence_len = 70  # 最大句子长度
    with open('./origin_data/' + file, encoding='utf-8') as f:
        for index, line in enumerate(f):
            content = line.strip().split('#')
            entity = content[0]
            att = content[1]

            relation_id = relation2id[content[2]]
            label = [0 for _ in range(len(relation2id))]
            label[relation_id] = 1

            # 处理句子特征
            output = []
            sentence = content[3]
            entity_pos = sentence.find(entity)  # 实体1位置
            if entity_pos == -1:
                entity_pos = 0
            att_pos = sentence.find(att)  # 实体2位置
            if att_pos == -1:
                att_pos = 0
            for i in range(fix_sentence_len):
                word = word2id['BLANK']
                relative_pos1 = pos_embed(i - entity_pos)  # 相对位置1
                relative_pos2 = pos_embed(i - att_pos)  # 相对位置2
                output.append([word, relative_pos1, relative_pos2])
            for i in range(min(fix_sentence_len, len(sentence))):
                word = 0
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]
                output[i][0] = word  # 替换成字符id

            train_x.append(output)
            train_y.append(label)
    return train_x, train_y


def read_data():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    with open('./origin_data/vec.txt', encoding='utf-8') as f:
        print("Loading word2vec file...")
        word_vector_dim = 100
        f.readline()  # 去除第一行
        for index, line in enumerate(f):
            content = line.strip().split()
            word2id[content[0]] = index
            word_vector = [float(i) for i in content[1:]]
            vec.append(word_vector)
        word2id['UNK'] = len(word2id)  # 如果出现word2vec中没有的字，则使用默认的“UNK”
        vec.append(np.random.normal(size=word_vector_dim, loc=0, scale=0.5))
        word2id['BLANK'] = len(word2id)  # 空白符
        vec.append(np.random.normal(size=word_vector_dim, loc=0, scale=0.5))
        vec = np.array(vec, dtype=np.float32)

    relation2id = {}
    with open('./origin_data/relation2id.txt', encoding='utf-8') as f:
        print("Load relation2id file...")
        for index, line in enumerate(f):
            content = line.strip().split()
            relation2id[content[0]] = index

    print("Reading train and test data...")
    x, y = data2array("train.txt", word2id, relation2id)
    train_x, dev_x, train_y, dev_y = train_test_split(x, y, test_size=0.05)
    test_x, test_y = data2array("test.txt", word2id, relation2id)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/dev_x.npy', dev_x)
    np.save('./data/dev_y.npy', dev_y)
    np.save('./data/test_x.npy', test_x)
    np.save('./data/test_y.npy', test_y)


def seperate():
    print('seperating train data')
    train_x = np.load('./data/train_x.npy')
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for sentence_feature in train_x:
        word_feature = []
        pos1_feature = []
        pos2_feature = []
        for [word, pos1, pos2] in sentence_feature:
            word_feature.append(word)
            pos1_feature.append(pos1)
            pos2_feature.append(pos2)
        train_word.append(word_feature)
        train_pos1.append(pos1_feature)
        train_pos2.append(pos2_feature)

    np.save('./data/train_word.npy', np.array(train_word))
    np.save('./data/train_pos1.npy', np.array(train_pos1))
    np.save('./data/train_pos2.npy', np.array(train_pos2))

    print('seperating dev data')
    train_x = np.load('./data/dev_x.npy')
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for sentence_feature in train_x:
        word_feature = []
        pos1_feature = []
        pos2_feature = []
        for [word, pos1, pos2] in sentence_feature:
            word_feature.append(word)
            pos1_feature.append(pos1)
            pos2_feature.append(pos2)
        train_word.append(word_feature)
        train_pos1.append(pos1_feature)
        train_pos2.append(pos2_feature)

    np.save('./data/dev_word.npy', np.array(train_word))
    np.save('./data/dev_pos1.npy', np.array(train_pos1))
    np.save('./data/dev_pos2.npy', np.array(train_pos2))

    print('seperating test data')
    test_x = np.load('./data/test_x.npy')
    test_word = []
    test_pos1 = []
    test_pos2 = []
    for sentence_feature in test_x:
        word_feature = []
        pos1_feature = []
        pos2_feature = []
        for [word, pos1, pos2] in sentence_feature:
            word_feature.append(word)
            pos1_feature.append(pos1)
            pos2_feature.append(pos2)
        test_word.append(word_feature)
        test_pos1.append(pos1_feature)
        test_pos2.append(pos2_feature)

    np.save('./data/test_word.npy', np.array(test_word))
    np.save('./data/test_pos1.npy', np.array(test_pos1))
    np.save('./data/test_pos2.npy', np.array(test_pos2))


read_data()
seperate()
