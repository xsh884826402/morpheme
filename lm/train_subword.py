from word2vec import Word2Vec_subword
from data_process import load_data_subword
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Config():
    def __init__(self):
        #
        self.original_file ='../data/text8'
        self.vocab_file = None
        self.split_ratio = [0.8, 0.1, 0.1]
        self.if_train = True
        self.if_test = False
        self.if_extend = False
        self.model_path = '../data/lm_model_subword/'

        self.batch_size = 1024
        self.vocabulary_size = 50000
        self.embedding_size = 256
        self.skip_window = 2
        self.max_epochs = 30
        self.num_skip = None
        self.num_sampled = 64


def main():

    config = Config()
    model = Word2Vec_subword(config)

    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True) ##每个gpu占用0.8                                                                              的显存
    tf_config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,)

    with tf.Session(config=tf_config) as sess:
        if config.if_train:
            x_train, y_train = load_data_subword(config.original_file, vocab_size=config.vocabulary_size, window_size=config.skip_window)
            init=tf.global_variables_initializer()
            sess.run(init)
            # word2id_path = os.path.join(os.path.dirname(config.original_file), 'word2id_dict_'+str(config.vocab_size)+'_winsize'+str(config.skip_window)+'_subword')
            word2id_path = os.path.join(os.path.dirname(config.original_file),
                                        'word2id_dict_' + str(config.vocabulary_size) + '_winsize' + str(
                                            config.skip_window) + '_subword')
            if not os.path.exists(word2id_path):
                print("Not found the word2id_dict")

            model.train_epoch(sess, x_train, y_train, config.model_path, word2id_path)

if __name__ == '__main__':
    main()



