# coding: utf-8
import os
import tensorflow as tf
from lstm_xsh import LSTM_Dynamic, train_epoch,Config
from data_process import train_dev_split,prepare_and_write_modelinputs
from data_process import load_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config(object):
    def __init__(self):
        self.batch_size = 1024
        self.hidden_size = 64
        self.vocab_size = 26
        self.embed_size = 500
        self.max_epochs = 40
        self.label_kinds = 2
        self.if_train = True
        self.if_test = False
        self.is_biLSTM = True
        self.max_seqlen = 20

        self.original_file = '../data/input_word_label.txt'
        self.train_file = '../data/train.txt'
        self.dev_file = '../data/dev.txt'
        self.vocab_file = '../data/vocab.txt'
        self.model_path = '/models/bilstm_500/'

        self.split_ratio = 0.8


def main():
    config = Config()
    config.if_train = True
    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')

    rnn_model = LSTM_Dynamic(config)
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True) ##每个gpu占用0.8 的显存
    tf_config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    with tf.Session(config=tf_config) as sess:
        if config.if_train:
            model_input_path = 'model_input' + str(config.embed_size) + os.path.basename(config.train_file)
            if not os.path.exists(model_input_path):
                prepare_and_write_modelinputs(config.train_file, model_input_path)
            train_epoch(rnn_model, sess, model_input_path, config.model_path)


    # print('Success for preparing data')


if __name__ == '__main__':
    main()
