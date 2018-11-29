# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import hyperparams
import transformer_network
import datetime

FLAGS = tf.flags.FLAGS


def main(_):
    save_path = './model/'
    word_embedding = np.load('./data/vec.npy')
    train_y = np.load('./data/train_y.npy')
    train_word = np.load('./data/train_word.npy')
    train_pos1 = np.load('./data/train_pos1.npy')
    train_pos2 = np.load('./data/train_pos2.npy')

    dev_y = np.load('./data/dev_y.npy')
    dev_word = np.load('./data/dev_word.npy')
    dev_pos1 = np.load('./data/dev_pos1.npy')
    dev_pos2 = np.load('./data/dev_pos2.npy')

    hp = hyperparams.Hyperparams()
    # settings.vocab_size = len(word_embedding)
    # settings.embedding_dim = len(word_embedding[0])
    hp.num_classes = len(train_y[0])
    hp.max_len = 70

    with tf.Graph().as_default() as g:
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            network = transformer_network.Transformer(char_embedding=word_embedding, hp=hp, is_training=True)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

        train_op = optimizer.minimize(network.final_loss, global_step=global_step)
        saver = tf.train.Saver(max_to_keep=None)

        merged_summary = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}

                feed_dict[network.input_word] = word_batch
                feed_dict[network.input_pos1] = pos1_batch
                feed_dict[network.input_pos2] = pos2_batch
                feed_dict[network.input_y] = y_batch

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, merged_summary, network.final_loss, network.accuracy], feed_dict)

                if step % 50 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {}".format(time_str, step, loss, accuracy)
                    print(tempstr)

            def dev_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}

                feed_dict[network.input_word] = word_batch
                feed_dict[network.input_pos1] = pos1_batch
                feed_dict[network.input_pos2] = pos2_batch
                feed_dict[network.input_y] = y_batch

                preds, loss, accuracy = sess.run(
                    [network.preds, network.final_loss, network.accuracy], feed_dict)

                return preds, loss, accuracy

            def prepare_data(num_steps, tmp_order, word_total, y_total, pos1_total, pos2_total):
                for i in range(num_steps):
                    start_index = i*hp.batch_size
                    end_index = min((i+1)*hp.batch_size, len(word_total))
                    batch_word = []
                    batch_pos1 = []
                    batch_pos2 = []
                    batch_y = []

                    temp = tmp_order[start_index:end_index]
                    for index in temp:
                        batch_word.append(word_total[index])
                        batch_pos1.append(pos1_total[index])
                        batch_pos2.append(pos2_total[index])
                        batch_y.append(y_total[index])

                    batch_word = np.array(batch_word)
                    batch_pos1 = np.array(batch_pos1)
                    batch_pos2 = np.array(batch_pos2)
                    batch_y = np.array(batch_y)
                    yield batch_word, batch_pos1, batch_pos2, batch_y

            train_order = list(range(len(train_word)))
            dev_order = list(range(len(dev_word)))
            cnt = 0
            best_f1 = 0.0
            for epoch in range(hp.num_epochs):
                np.random.shuffle(train_order)
                num_train_steps = int((len(train_word)-1) / float(hp.batch_size)) + 1
                for batch_word, batch_pos1, batch_pos2, batch_y in prepare_data(num_train_steps, train_order,
                                                                                train_word, train_y,
                                                                                train_pos1, train_pos2):
                    train_step(batch_word, batch_pos1, batch_pos2, batch_y)

                preds = list()
                num_dev_steps = int((len(dev_word)-1) / float(hp.batch_size)) + 1
                for batch_word, batch_pos1, batch_pos2, batch_y in prepare_data(num_dev_steps, dev_order,
                                                                                dev_word, dev_y,
                                                                                dev_pos1, dev_pos2):
                    pred, _, _ = dev_step(batch_word, batch_pos1, batch_pos2, batch_y)
                    preds.extend(pred)

                cnt_y = cnt_pre = cnt_all = 0
                for item_y, item_pre in zip(dev_y, preds):
                    if item_y[1] == 1:
                        cnt_y += 1
                    if item_pre == 1:
                        cnt_pre += 1
                    if item_y[1] == 1 and item_pre == 1:
                        cnt_all += 1
                if cnt_pre == 0:
                    print("Epoch: {}".format(epoch))
                    print("all 0")
                else:
                    precision = float(cnt_all / cnt_pre)
                    recall = float(cnt_all / cnt_y)
                    f1 = 2 * precision * recall / (precision + recall)
                    print("Epoch: {}".format(epoch))
                    print("precision: %.4f " % precision, "cnt_pre: %d " % cnt_pre)
                    print("recall: %.4f " % recall, "cnt_y: %d " % cnt_y)
                    print("f1: %.4f " % f1, "true: %d " % cnt_all)

                if epoch < 10:
                    continue
                elif f1 > best_f1:
                    best_f1 = f1
                    cnt = 0
                    print("best f1, Epoch " + str(epoch) + ": model saved!")
                    saver.save(sess, save_path + 'transformer_model')
                else:
                    cnt += 1
                    if cnt >= 5:
                        break


if __name__ == "__main__":
    tf.app.run()
