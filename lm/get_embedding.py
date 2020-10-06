from word2vec import Word2Vec
from data_process import build_data
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Config():
    def __init__(self):
        #
        self.original_file = '../data/text8'
        self.vocab_file = None
        self.split_ratio = [0.8, 0.1, 0.1]
        self.if_train = True
        self.if_test = False
        self.if_extend = True
        self.model_path = '../data/lm_model/'

        self.batch_size = 1024
        self.vocabulary_size = 50000
        self.embedding_size = 256
        self.skip_window = 2
        self.max_epochs = 60
        self.num_skip = None
        self.num_sampled = 64


def main():
    config = Config()
    model = Word2Vec(config)
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True) ##每个gpu占用0.8                                                                              的显存
    tf_config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,)
    with tf.Session(config=tf_config) as sess:
        if config.if_train:
            outfile = os.path.join(os.path.dirname(config.original_file),
                                   'training_data_' + str(config.vocabulary_size) + '_winsize' + str(config.skip_window))
            if not os.path.exists(outfile):
                build_data(config.original_file, outfile, config.vocabulary_size, config.skip_window)
            init=tf.global_variables_initializer()
            sess.run(init)

            word2id_path = os.path.join(os.path.dirname(config.original_file),
                                        'word2id_dict_' + str(config.vocabulary_size) + '_winsize' + str(
                                            config.skip_window))
            if not os.path.exists(word2id_path):
                print("Not found the word2id_dict")
            model_name ='../data/lm_model/models_epoch30'
            model.predict_epoch(sess, outfile, config.model_path, word2id_path,model_name)


if __name__ == '__main__':
    main()



