# coding: utf-8
import os
import tensorflow as tf
from lstm_xsh import LSTM_Dynamic, infer, Config
from data_process import train_dev_split, prepare_modelinputs

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 每次跑测试需要更改test_path,显卡



def inference_main(contents):
    '''

    :param contents: 一维列表，每行是一个字符串
    :return:
    '''
    config = Config()
    config.batch_size = 10000
    config.if_inference = True
    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')
    blstm_model = LSTM_Dynamic(config)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        if config.if_inference:

            x_infer  = prepare_modelinputs(contents,y_flag=False)
            if len(x_infer) < config.batch_size:
                for i in range(0, config.batch_size - len(x_infer)):
                    x_infer.append([0])
            print('Target to special model to test')
            test_model = os.path.join(config.model_path, "models_epoch39")
            print('Start do predicting...','\n'*10)
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, test_model)
            y_pred = infer(blstm_model, sess, x_infer, list(map(len, x_infer)))
            return y_pred


def build_word_subword_txt(infile, outfile):
    contents = []
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            contents.extend(line)
    y_pred = inference_main(contents)
    all_subwords = []
    # prepare subwords
    for x, y in zip(contents, y_pred):
        # assert len(x)==len(y)
        subwords = []
        left_index = -1
        for i, l in enumerate(y):
            if l==1:
                if left_index==-1:
                    subwords.append(x[:i])
                else:
                    subwords.append(x[left_index:i])
                left_index = i
        if left_index != -1:
            subwords.append(x[left_index:])
        if len(subwords)>3:
            process_subword =[subwords[0]]+["".join(subwords[1:-1])]+[subwords[-1]]
        elif len(subwords)==3:
            process_subword = subwords
        elif len(subwords)==2:
            process_subword = [subwords[0]] + [''] + [subwords[1]]
        elif len(subwords)==1:
            process_subword = [subwords[0]] + [''] + ['']
        else:
            process_subword = ['', '', '']
        process_subword = [item  if item != '' else '#' for item in process_subword]
        if len(process_subword) == 3:
            all_subwords.append(process_subword)
        else:
            print('x,y,',x,y)

    # writing results
    with open(outfile, 'w') as f:
        for word, subwords in zip(contents, all_subwords):
            f.write(word+'\t'+" ".join(subwords)+'\n')
    print('Building word subword text Successfully')

if __name__ == '__main__':
    infile = '../data/text8'
    outfile = '../data/subword_using_model.txt'
    build_word_subword_txt(infile, outfile)