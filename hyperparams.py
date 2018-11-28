# -*- coding: utf-8 -*-
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''


class Hyperparams:
    '''Hyperparameters'''
    
    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory
    
    # model
    maxlen = 70 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    #  alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 25
    num_heads = 8
    num_classes = 2
    char_dim = 512
    pos_dim = 512
    pos_num = 143
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    hidden_units = 512
