import os
import argparse
if __name__ == '__main__':
    vectors = ['infer_word_vector_8', 'bpe_word_vector_8', 'raw_word_vector_8']
    for vector in vectors:
        new_cmd = 'python word_analogy.py --vector '+"../data/Visual_vector/"+vector
        for n in range(100):
            if n%2==0:
                new_cmd_1 = new_cmd + " --topn "+str(n)
                os.system(new_cmd_1)
