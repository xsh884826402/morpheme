# coding: utf-8
import os
import tensorflow as tf
from lstm_xsh import LSTM_Dynamic,test,Config
from data_process import train_dev_split, prepare_modelinputs
from argparse import ArgumentParser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 每次跑测试需要更改test_path



def main(embed_size, test_model_path):
    config = Config()
    config.if_test = True
    config.embed_size = embed_size
    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')

    blstm_model = LSTM_Dynamic(config)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        if config.if_test:

            with open(config.dev_file) as f:
                contents = f.readlines()
            x_test , y_test = prepare_modelinputs(contents)
            if len(x_test) < config.batch_size:
                for i in range(0, config.batch_size - len(x_test)):
                    x_test.append([0])
                    y_test.append([0])

            print('Target to special model to test')
            print("Using model {}".format(test_model_path))
            # test_model = os.path.join(config.model_path, "models_epoch1")
            print('Start do predicting...')
            saver = tf.train.Saver(tf.trainable_variables())
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, test_model_path)
            test(blstm_model, sess, x_test, y_test, list(map(len, x_test)),test_model_path)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--embed_size')
    parser.add_argument('--test_model_path')
    args =parser.parse_args()
    main(int(args.embed_size), args.test_model_path)
