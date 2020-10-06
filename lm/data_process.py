import os
import sys
import collections
import traceback
import time
import pickle
import numpy as np
import argparse
def load_data(infile, vocab_size = 50000,window_size=2, ):
    outfile =os.path.join(os.path.dirname(infile), 'training_data_'+str(vocab_size)+'_winsize'+str(window_size))
    if not os.path.exists(outfile):
        build_data(infile, outfile,vocab_size, window_size)
    # print('start reading data')
    # x, y = read_data(outfile)
    # print('finish reading data')
    # return x, y

def build_data(infile, outfile, vocab_size, window_size):
    '''

    :param infile: 输入的训练数据
    :param outfile: 输出的装换成id的训练数据
    :param vocab_size: 词汇表大小
    :param window_size: 滑动窗口的大小
    :return:
    '''
    with open(infile, 'r') as f:
        line = f.readline()
        line = line.strip().split()

    count = [['UNK', -1]]
    count.extend(collections.Counter(line).most_common(vocab_size-1))
    word2id = dict()
    id_index = 0
    for word, _ in count:
        word2id[word] = id_index
        id_index += 1
    id2word =dict(zip(word2id.values(),word2id.keys()))
    # 构造词向量

    try:
        with open(outfile, 'w') as f:
            for i in range(window_size, len(line)-window_size):
                x = line[i]
                if x not in word2id:
                    continue
                for j in range(-window_size,window_size+1):
                    if j==0:
                        continue
                    else:
                        if line[i+j] in word2id:
                            f.write(str(word2id[x])+'\t'+str(word2id[line[i+j]])+'\n')
                        else:
                            f.write(str(word2id[x])+'\t'+str(word2id['UNK'])+'\n')
        word2id_dict_path = os.path.join(os.path.dirname(outfile),
                                         'word2id_dict_' + os.path.basename(outfile))
        with open(word2id_dict_path, 'w') as f:
            for word, id in word2id.items():
                f.write(str(word) + '\t' + str(id) + '\n')
        print('Build data successful')
    except :
        traceback.print_exc()
        print('Delete '+outfile)
        os.remove(outfile)
    return

def build_data_subword(infile, outfile, vocab_size, window_size, subword_dict_path ='../data/subword.txt'):
    # load subword_dict
    # subword_dict_path
    # word '\t' subword ' ' subword ' ' subword ' '
    start_time = time.time()
    word_subwords_dict = dict()
    with open(subword_dict_path, 'r') as f:
        for line in f:
            word, subwords = line.strip().split('\t')
            subwords = subwords.split()
            word_subwords_dict[word] = subwords
    word_subwords_time = time.time()
    print('Build word_subword_dict using time:'+str(word_subwords_time-start_time))



    words_count_dict = collections.defaultdict(int)
    with open(infile, 'r') as f:
        input_line = f.readline()
        input_line = input_line.strip().split()
    for word in input_line:
        words_count_dict[word] += 1

    words_subwords_count_dict = collections.defaultdict(int)
    for word in words_count_dict:
        n = words_count_dict[word]
        words_subwords_count_dict[word] += n
        for subword in word_subwords_dict[word]:
            words_subwords_count_dict[subword] += n

    words_count_time = time.time()
    print('BUild words_count_dict using time:'+str(words_count_time-word_subwords_time))

    reversed_words_subowrds_count_dict = collections.defaultdict(list)
    for word,count in words_subwords_count_dict.items():
        reversed_words_subowrds_count_dict[count].append(word)

    keys = sorted(reversed_words_subowrds_count_dict.keys(), reverse=True)

    count = [['UNK', -1]]
    count_len = 1
    for key in keys:
        if count_len <vocab_size:
            count_len += len(reversed_words_subowrds_count_dict[key])
            for word in reversed_words_subowrds_count_dict[key]:
                count.append([word, key])
    count = count[:vocab_size]

    word2id = dict()
    id_index = 0
    for word, _ in count:
        word2id[word] = id_index
        id_index += 1
    id2word = dict(zip(word2id.values(), word2id.keys()))
    word2id_time = time.time()
    print('Build word2id_dict time:'+str(word2id_time-words_count_time))

    # word_subwords_dict
    # key:enhance value en  # ce

    try:
        with open(outfile, 'w') as f:
            for i in range(window_size, len(input_line)-window_size):
                x = input_line[i]
                if x not in word2id:
                    continue
                for j in range(-window_size,window_size+1):
                    if j==0:
                        continue
                    else:
                        word_subword_str = [word2id[x]] + [word2id[item] if item in word2id else word2id['UNK'] for item
                                                           in word_subwords_dict[x]]
                        word_subword_str = " ".join(list(map(str,word_subword_str)))
                        if input_line[i+j] in word2id:
                            f.write(word_subword_str+'\t'+str(word2id[input_line[i+j]])+'\n')
                        else:
                            f.write(word_subword_str+'\t'+str(word2id['UNK'])+'\n')
        write_outfile_time = time.time()
        print('Write outfile using time :{}'.format(write_outfile_time-word2id_time))
        word2id_dict_path = os.path.join(os.path.dirname(outfile), 'word2id_dict_'+str(vocab_size)+'_winsize'+str(window_size)+'_subword')
        with open(word2id_dict_path, 'w') as f:
            for word, id in word2id.items():
                f.write(str(word)+'\t'+str(id)+'\n')
        print('Build data successful')
        return True
    except :
        traceback.print_exc()
        print('Delete '+outfile)
        os.remove(outfile)
        return False

def build_data_subword_extend(infile, outfile, vocab_size, window_size, subword_dict_path ='../data/subword.txt',subword_size_ratio = 4):
    # 词汇表扩展而来，total_vocab_size = vocab_size 和 subword_size
    # load subword_dict
    # subword_dict_path
    # word '\t' subword ' ' subword ' ' subword ' '
    start_time = time.time()
    word_subwords_dict = collections.defaultdict(list)
    with open(subword_dict_path, 'r') as f:
        for line in f:
            word, subwords = line.strip().split('\t')
            subwords = subwords.split()
            word_subwords_dict[word] = subwords
    word_subwords_time = time.time()
    print('Build word_subwords_dict from {}'.format(subword_dict_path))
    print('Build word_subword_dict using time:'+str(word_subwords_time-start_time))

    with open(infile, 'r') as f:
        line = f.readline()
        line = line.strip().split()

    word2id_keys = []
    count = [['UNK', -1]]
    word_counter = collections.Counter(line).most_common(vocab_size-1)
    count.extend(word_counter)
    word2id = dict()
    id_index = 0
    for word, _ in count:
        word2id[word] = id_index
        word2id_keys.append(word)
        id_index += 1


    subwords_list = []
    for word, n in word_counter:
        if word in word_subwords_dict.keys():
            for _ in range(n):
                subwords_list.extend(word_subwords_dict[word])

    subwords_vocab_size = vocab_size//subword_size_ratio
    subword_counter = collections.Counter(subwords_list).most_common(len(subwords_list))
    print('Lenth of subwords: {}'.format(len(subword_counter)), '\n'*10)
    for word, _  in subword_counter:
        if id_index>= vocab_size+subwords_vocab_size:
            break
        if word not in word2id.keys():
            word2id[word] = id_index
            word2id_keys.append(word)
            id_index += 1

    words_count_time = time.time()
    print('BUild words Counter'+str(words_count_time-word_subwords_time))




    id2word = dict(zip(word2id.values(), word2id.keys()))
    word2id_time = time.time()
    print('Build word2id_dict time:'+str(word2id_time-words_count_time))

    # word_subwords_dict
    # key:enhance value en  # ce

    try:
        with open(outfile, 'w') as f:
            for i in range(window_size, len(line)-window_size):
                x = line[i]
                if x not in word2id:
                    continue
                for j in range(-window_size,window_size+1):
                    if j==0:
                        continue
                    else:
                        word_subword_str = [word2id[x]] + [word2id[item] if item in word2id else word2id['UNK'] for item
                                                           in word_subwords_dict[x]]
                        word_subword_str = " ".join(list(map(str,word_subword_str)))
                        if line[i+j] in word2id:
                            f.write(word_subword_str+'\t'+str(word2id[line[i+j]])+'\n')
                        else:
                            f.write(word_subword_str+'\t'+str(word2id['UNK'])+'\n')
        write_outfile_time = time.time()
        print('Write outfile using time :{}'.format(write_outfile_time-word2id_time))
        word2id_dict_path = os.path.join(os.path.dirname(outfile), 'word2id_dict_'+os.path.basename(outfile))
        with open(word2id_dict_path, 'w') as f:
            for word in word2id_keys:
                f.write(str(word)+'\t'+str(word2id[word])+'\n')
        print('Build data successful')
        return True
    except :
        traceback.print_exc()
        print('Delete '+outfile)
        os.remove(outfile)
        return False

def read_data(infile,):
    _allx = []
    _ally = []
    with open(infile, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            _allx.append(x)
            _ally.append(y)
    return _allx, _ally
def read_data_subword(infile):
    all_x = []
    all_y = []
    with open(infile, 'r') as f:
        for line in f:
            x, y = line.strip().split('\t')
            all_x.append(x.split(' '))
            all_y.append(y)
    return all_x, all_y

def load_data_subword(infile, vocab_size = 50000, window_size = 2):
    outfile = os.path.join(os.path.dirname(infile), 'training_data_'+str(vocab_size)+'_winsize'+str(window_size)+'_subword_extend')
    if not os.path.exists(outfile):
        flag = build_data_subword_extend(infile, outfile, vocab_size, window_size)


def convert_ndarry_to_vector(word2id_path, array_path='../data/lm_model/embedding/29', outfile ='../data/lm_model/embedding/word_embedding29'):
    word2id = {}
    with open(word2id_path, 'r') as f:
        for line in f:
            word, id =line.strip().split()
            word2id[word] = id
    with open(array_path, 'rb') as f:
        matrix = pickle.load(f)
    with open(outfile, 'w') as f:
        for word in word2id.keys():
            f.write(word+'\t'+' '.join(list(map(str, matrix[int(word2id[word])])))+'\n')
    print('convert_ndarry_to_word_vector Successfully')

def test_split(infile, ratio= 0.2,):
    with open(infile, 'r') as f:
        words = f.readline().strip().split()
    words = words[:int(len(words)*ratio)]
    outfile = infile+'_test_code'
    with open(outfile, 'w') as f:
        f.write(" ".join(words)+'\n')
    print('Test split Successfully ')

def merge_subwords_to_words(infile, outfile, word_subwords_dict,embedding_size = 256,mode=1):

    word_embedding_dict = dict()
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            # assert len(line) == (embedding_size+1)
            word = line[0]
            embedding = list(map(float, line[1:]))
            word_embedding_dict[word] = embedding
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
        print('outfile not exists,build directory')
    with open(outfile, 'w') as fw:
        for word in word_embedding_dict.keys():
            temp_embedding = [word_embedding_dict[word]]
            for subword in word_subwords_dict[word]:
                if mode==1:
                    if subword in word_embedding_dict.keys():
                        temp_embedding.append(word_embedding_dict[subword])
                if mode==2:
                    if subword in word_embedding_dict.keys():
                        temp_embedding.append(word_embedding_dict[subword])
                    else:
                        temp_embedding.append(word_embedding_dict['UNK'])
            final_embedding = []
            word_count = len(temp_embedding)
            temp_array = np.array(temp_embedding)
            temp_array = np.sum(temp_array, axis=0)
            temp_array = temp_array / word_count
            temp_str = " ".join(list(map(str,temp_array.tolist())))
            fw.write(word + '\t'+temp_str+'\n')
    print('merge subword successful')
    print('Storing new word_vector in {} :'.format(outfile))


def concat_subwords_to_words(infile, outfile, word_subwords_dict,embedding_size = 256,mode=1):

    word_embedding_dict = dict()
    print('straing concat from infile {}'.format(infile))
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            # assert len(line) == (embedding_size+1)
            word = line[0]
            embedding = line[1:]
            word_embedding_dict[word] = embedding
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
        print('outfile not exists,build directory')
    with open(outfile, 'w') as fw:
        for word in word_embedding_dict.keys():
            # if word=='was':
            #     print('len was',len(word_embedding_dict['was']))
            #     print('len wa',len(word_embedding_dict['wa']))
            #     print('len #',len(word_embedding_dict['#']))
            #     print('len s', len(word_embedding_dict['s']))
            #     print('len UNK', word_embedding_dict['UNK'])

            # 注意这里要构建一个新的list，不然会改变原来词典的值
            temp_embedding = list(word_embedding_dict[word])
            if word in word_subwords_dict.keys():
                assert len(word_subwords_dict[word])==3
                for subword in word_subwords_dict[word]:
                    if subword in word_embedding_dict.keys():
                        temp_embedding.extend(word_embedding_dict[subword])
                    else:
                        temp_embedding.extend(word_embedding_dict['UNK'])
            else:
                # print('word not in wors_subwords_dict', word)
                temp_embedding = temp_embedding*4

            # print('temp_embedding {}'.format(len(temp_embedding)))
            # assert len(word_embedding_dict[word])*4 ==len(temp_embedding)

            temp_str = " ".join(temp_embedding)
            try:
                assert len(temp_embedding)==embedding_size*4
            except:
                print('Error word', word,len(temp_embedding), word_subwords_dict[word])
                return
            # print('length ',len(temp_embedding))
            fw.write(word + '\t'+temp_str+'\n')
    print('Concat subword successful')
    print('Storing new word_vector in {} :'.format(outfile))

def merge_indir_outdir(input_directory, output_directory, word_subword_file,mode=1):
    word_subwords_dict = collections.defaultdict(list)
    with open(word_subword_file, 'r') as f:
        for line in f:
            word, subwords = line.strip().split('\t')
            subwords = subwords.split()
            word_subwords_dict[word] = subwords
    print('Loading word_subword successful from {}'.format(word_subword_file))
    for file in os.listdir(input_directory):
        if os.path.exists(output_directory + file):
            print('file exists {}'.format(output_directory+file))
            continue
        merge_subwords_to_words(input_directory + file, output_directory + file, word_subwords_dict, mode=mode)
    print('Mergeing vector from indir to outdir')

def concat_indir_outdir(input_directory, output_directory, word_subword_file,mode=1):
    print('Concating')
    word_subwords_dict = collections.defaultdict(list)
    with open(word_subword_file, 'r') as f:
        for line in f:
            word, subwords = line.strip().split('\t')
            subwords = subwords.split()
            word_subwords_dict[word] = subwords
    print('Loading subword from {} Successfully '.format(word_subword_file))
    for file in os.listdir(input_directory):
        if os.path.exists(output_directory + file):
            continue
        concat_subwords_to_words(input_directory + file, output_directory + file, word_subwords_dict, mode=mode)
    print('Concating vector from indir to outdir')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_flag', default=False)
    parser.add_argument('--input_dir')
    parser.add_argument('--merge_dir')
    parser.add_argument('--concat_dir')
    parser.add_argument('--subword_path')
    parser.add_argument('--concat_flag', default=False)
    args = parser.parse_args()
    print(type(args.concat_flag))
    print(type(args.merge_flag))
    if args.merge_flag== 'True':
        print('Staring merge')
        input_directory = args.input_dir
        merge_directory = args.merge_dir
        word_subword_path = args.subword_path
        merge_indir_outdir(input_directory, merge_directory, word_subword_file=word_subword_path, mode=2)
    if args.concat_flag == 'True':
        print('Starting concat')
        input_directory = args.input_dir
        concat_directory = args.concat_dir
        word_subword_path = args.subword_path
        concat_indir_outdir(input_directory, concat_directory, word_subword_file=word_subword_path)


    #注意修改 vector_directory,result_path
    # suword merges
    # root_path ='/home/ubuntu/User/xsh/Paper/'

    # input_directory = '../data/lm_model_subword_extend_using_inference_ws2_oneshuffle_6_3/embedding/'
    # output_directory = '../data/lm_model_subword_extend_using_inference_ws2_oneshuffle_6_3/merge_embedding_ensure3subword/'
    # word_subword_path = '../data/subword_using_model.txt'
    # merge_indir_outdir(input_directory, output_directory,word_subword_path,mode=2)

    # vector_directory_path = '/tmp-data/xushenghua/practice/data/lm_model_subword_extend_ws2/merge_embedding/'
    # similarity_py_path = '/tmp-data/xushenghua/practice/en_embedding_similarity/all_wordsim.py'
    # similarity_data_path = '/tmp-data/xushenghua/practice/en_embedding_similarity/data/en'
    # result_path ='/tmp-data/xushenghua/practice/lm/lm_extend_ws2_result.txt'
    # vector_directory_path =args.merge_dir+'merge_embeddingword_vector_'
    # similarity_py_path =root_path + 'en_embedding_similarity/all_wordsim.py'
    # similarity_data_path = root_path + 'en_embedding_similarity/data/en'
    # result_path = root_path + 'lm/lm_model_subword_extend_using_bpe_result.txt'
    # max_epoch = 60
    # for file_index in range(max_epoch):
    #
    #     if file_index%2 != 0:
    #         continue
    #
    #     new_cmd = 'python '+ similarity_py_path + ' --vector=' +vector_directory_path+str(file_index)+' --similarity='+ similarity_data_path +' --result_file='+result_path
    #     os.system(new_cmd)



