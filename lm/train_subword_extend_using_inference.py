# 每次更新 显卡，self.model_path
from word2vec import Word2Vec_subword_extend,train_epoch
from data_process import build_data_subword_extend
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class Config():
    def __init__(self):
        #
        self.original_file ='../data/text8'
        self.vocab_file = None
        self.split_ratio = [0.8, 0.1, 0.1]
        self.if_train = True
        self.if_test = False
        self.if_extend = True
        self.model_path = '../data/lm_model_subword_extend_using_inference_ws2_Multishuffle/'

        self.batch_size = 1024
        self.vocabulary_size = 50000
        self.embedding_size = 256
        self.skip_window = 2
        self.max_epochs = 30
        self.num_skip = None
        self.num_sampled = 64


def main():

    config = Config()
    model = Word2Vec_subword_extend(config)

    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    tf_config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,)

    with tf.Session(config=tf_config) as sess:
        if config.if_train:
            outfile = os.path.join(os.path.dirname(config.original_file), 'training_data_' + str(config.vocabulary_size) + '_winsize' + str(
                config.skip_window) + '_subword_extend_using_inference')
            subword_dict_path = '../data/subword_using_model.txt'
            build_data_subword_extend(config.original_file, outfile, config.vocabulary_size, config.skip_window, subword_dict_path=subword_dict_path)
            init=tf.global_variables_initializer()
            sess.run(init)
            word2id_path = os.path.join(os.path.dirname(config.original_file),
                                        'word2id_dict_'+os.path.basename(outfile))
            if not os.path.exists(word2id_path):
                print("Not found the word2id_dict")
                return

            train_epoch(model, sess, outfile, config.model_path, word2id_path)


if __name__ == '__main__':
    main()



