# coding: utf-8
import os
import tensorflow as tf
from lstm_xsh import LSTM_Dynamic
from data_process import train_dev_split
from data_process import load_data
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Config(object):
    def __init__(self):
        self.batch_size = 1024
        self.hidden_size = 64
        self.vocab_size = 26
        self.embed_size = 512
        self.max_epochs = 40
        self.label_kinds = 2
        self.if_train = True
        self.if_test = False
        self.is_biLSTM = True
        self.max_seqlen = 20

        self.original_file = './data/input_word_label.txt'
        self.train_file = './data/train.txt'
        self.dev_file = './data/dev.txt'
        self.vocab_file = 'data/vocab.txt'
        self.model_path = 'models/bilstm/'

        self.split_ratio = 0.8


def main():
    config = Config()

    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')

    lstm_config = Config()
    rnn_model = LSTM_Dynamic(lstm_config)
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True) ##每个gpu占用0.8                                                                              的显存
    config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if lstm_config.if_train:
            init=tf.global_variables_initializer()
            sess.run(init)
            (X_train, y_train) = load_data(lstm_config.train_file)

            if len(X_train) < lstm_config.batch_size:
                for i in range(0, lstm_config.batch_size - len(X_train)):
                    X_train.append([0])
                    y_train.append([0])

            seq_len_train = list(map(lambda x: len(x), X_train))

            rnn_model.train_epoch(sess, lstm_config.train_file, X_train,
                                  y_train, seq_len_train,
                                  lstm_config.model_path)

    print('Success for preparing data')


if __name__ == '__main__':
    main()
