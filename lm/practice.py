import os
import collections
import numpy as np


def shuffle(infile, outfile):
    with open(infile, 'r') as f:
        contents = f.readlines()
        contents = [item.strip() for item in contents]
        np.random.shuffle(contents)
    with open(outfile, 'w') as fw:
        for content in contents:
            fw.write(content + '\n')
    print('Shuffle Successfully')


def generate_batch(infile, batchsize):
    if batchsize==0:
        print('batchsize must > 0')
        return None, None
    x_batch,y_batch = [], []
    count = 0
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            x = line[:-1]
            y = line[-1]
            x_batch.append(x)
            y_batch.append([y])
            count += 1
            if count==batchsize:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
                count = 0

if __name__ == '__main__':
    # file1 = '../data/subword.txt'
    # file2 = '../data/subword_using_model.txt'
    # resut_path ='../diff.txt'
    # with open(file1) as f1, open(file2) as f2:
    #     with open(resut_path, 'w') as f:
    #         for line1, line2 in zip(f1, f2):
    #             lien1 = line1.strip().split()
    #             line2 = line2.strip().split()
    #             if len(line1) != len(line2):
    #                 f.write('1:\t' + " ".join(line1) + '2:\t' + " ".join(line2) + '\n')
    #

    # file1 = '../data/lm_model_subword_extend_concat/concat_embedding/word_vector_1'
    # with open(file1, 'r') as f:
    #     for line in f:
    #
    #         line = line.strip().split()
    #         word = line[0]
    #         vector = line[1]
    #         print(len(vector))
    a = (3,4,5)
    b = list(a)
    print(b)
