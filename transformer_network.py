import tensorflow as tf
from hyperparams import Hyperparams as Hp
from modules import *


class Transformer(object):
    def __init__(self, char_embedding, hp: Hp, is_training=False):
        self.input_word = tf.placeholder(tf.int32, [None, hp.maxlen], name='input_word')
        self.input_pos1 = tf.placeholder(tf.int32, [None, hp.maxlen], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, [None, hp.maxlen], name='input_pos2')
        self.input_y = tf.placeholder(tf.int32, [None, hp.num_classes], name='true_labels')

        with tf.name_scope("embedding"):
            ## Embedding
            self.enc = embedding(self.input_word,
                                 vocab_size=len(char_embedding),
                                 num_units=hp.char_dim,
                                 scale=True,
                                 scope="enc_embed")

            self.enc += embedding(self.input_pos1,
                                  vocab_size=hp.pos_num,
                                  num_units=hp.pos_dim,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pos1")

            self.enc += embedding(self.input_pos2,
                                  vocab_size=hp.pos_num,
                                  num_units=hp.pos_dim,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pos2")

        with tf.name_scope("encoder"):
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=hp.hidden_units,
                                                   num_heads=hp.num_heads,
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training=is_training,
                                                   causality=False
                                                   )
                    self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

        with tf.name_scope("outputs"):
            self.enc = tf.reshape(self.enc, shape=[-1, hp.maxlen*hp.hidden_units])
            # self.enc = tf.layers.dense(self.enc, 200, activation=tf.nn.relu, use_bias=True,
            #                            kernel_initializer=tf.random_normal_initializer(-0.25, 0.25),
            #                            bias_initializer=tf.constant_initializer(0.01))
            self.logits = tf.layers.dense(self.enc, hp.num_classes, activation=None, use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(-0.25, 0.25),
                                          bias_initializer=tf.constant_initializer(0.01))
            self.pred_probability = tf.nn.softmax(self.logits, name="class_probability")
            self.preds = tf.argmax(self.pred_probability, axis=-1, name="class_prediction")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                  weights_list=tf.trainable_variables())
            self.final_loss = tf.reduce_mean(self.cross_entropy) + self.l2_loss
            tf.summary.scalar("loss", self.final_loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.preds, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
