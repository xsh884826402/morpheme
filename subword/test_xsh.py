# coding: utf-8
import os
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from lstm_xsh import LSTM_Dynamic
from data_process import train_dev_split
from data_process import load_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 每次跑测试需要更改test_path

class Config(object):
    def __init__(self):
        self.batch_size = 1024
        self.hidden_size = 64
        self.vocab_size = 26
        self.embed_size = 512
        self.max_epochs = 40
        self.label_kinds = 2
        self.if_train = False
        self.if_test = True
        self.is_biLSTM = True
        self.max_seqlen = 20


        self.original_file = 'data/zhwiki-latest-pages-articles.txt'
        self.train_file = 'data/train.txt'
        self.dev_file = 'data/dev.txt'
        self.vocab_file = 'data/vocab.txt'
        self.model_path = 'models/bilstm/'

        self.split_ratio = 0.9


def main():
    config = Config()
    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')
    lstm_config = Config()
    blstm_model = LSTM_Dynamic(lstm_config)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if lstm_config.if_test:
            print('Start load dev data ...')
            (X_test, y_test) = load_data(lstm_config.dev_file)
            print('Loading sucess!')

            if len(X_test) < lstm_config.batch_size:
                for i in range(0, lstm_config.batch_size - len(X_test)):
                    X_test.append([0])
                    y_test.append([0])

            seq_len_test = list(map(lambda x: len(x), X_test))
            print('Target to special model to test')
            test_model = os.path.join(lstm_config.model_path, "models_epoch38")
            print('Start do predicting...')
            blstm_model.test(sess, test_model, X_test, y_test, seq_len_test,
                             lstm_config.vocab_file, lstm_config.model_path+'result/')


if __name__ == '__main__':
    main()
