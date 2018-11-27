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

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, merged_summary, network.final_loss, network.accuracy], feed_dict)

                if step % 50 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {}".format(time_str, step, loss, accuracy)
                    print(tempstr)

            for epoch in range(hp.num_epochs):
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(hp.batch_size))):
                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * hp.batch_size: (i + 1) * hp.batch_size]
                    for index in temp_input:
                        temp_word.append(train_word[index])
                        temp_pos1.append(train_pos1[index])
                        temp_pos2.append(train_pos2[index])
                        temp_y.append(train_y[index])

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    train_step(temp_word, temp_pos1, temp_pos2, temp_y)
                print("epoch " + str(epoch) + ": model saved!")
                saver.save(sess, save_path + 'transformer_model')


if __name__ == "__main__":
    tf.app.run()
