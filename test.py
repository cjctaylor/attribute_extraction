# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import transformer_network
import hyperparams


def main(_):
    pathname = "./model/transformer_model"
    word_embedding = np.load("./data/vec.npy")
    test_y = np.load('./data/test_y.npy')
    test_word = np.load('./data/test_word.npy')
    test_pos1 = np.load('./data/test_pos1.npy')
    test_pos2 = np.load('./data/test_pos2.npy')
    test_hp = hyperparams.Hyperparams()
    test_hp.num_classes = len(test_y[0])
    test_hp.word_length = 70
    test_hp.batch_size = len(test_word)

    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            with tf.variable_scope("model"):
                network = transformer_network.Transformer(char_embedding=word_embedding, hp=test_hp, is_training=False)

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}
                word_list = []
                pos1_list = []
                pos2_list = []
                for i in range(len(word_batch)):
                    word_list.append([word for word in word_batch[i]])
                    pos1_list.append([pos for pos in pos1_batch[i]])
                    pos2_list.append([pos for pos in pos2_batch[i]])
                word_list = np.array(word_list)
                pos1_list = np.array(pos1_list)
                pos2_list = np.array(pos2_list)

                feed_dict[network.input_word] = word_list
                feed_dict[network.input_pos1] = pos1_list
                feed_dict[network.input_pos2] = pos2_list
                feed_dict[network.input_y] = y_batch
                accuracy, preds = sess.run([network.accuracy, network.preds], feed_dict)
                return accuracy, preds

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            accuracy, preds = test_step(test_word, test_pos1, test_pos2, test_y)
            print(accuracy)
            print(preds)

            cnt_y = cnt_pre = cnt_all = 0
            for item_y, item_pre in zip(test_y, preds):
                if item_y[1] == 1:
                    cnt_y += 1
                if item_pre == 1:
                    cnt_pre += 1
                if item_y[1] == 1 and item_pre == 1:
                    cnt_all += 1
            precision = float(cnt_all/cnt_pre)
            recall = float(cnt_all/cnt_y)
            f1 = 2*precision*recall/(precision+recall)
            print("precision: %.4f " % precision, "cnt_pre: %d " % cnt_pre)
            print("recall: %.4f " % recall, "cnt_y: %d " % cnt_y)
            print("f1: %.4f " % f1, "true: %d " % cnt_all)


if __name__ == '__main__':
    tf.app.run()
