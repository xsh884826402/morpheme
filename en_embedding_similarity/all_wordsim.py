import sys
import os

from optparse import OptionParser
from read_write import read_word_vectors
from ranking import *

if __name__=='__main__':
    parser = OptionParser()
    parser.add_option("--vector", dest="vector", help="vector file")
    parser.add_option("--similarity", dest="similarity", default="./data/en", help="similarity file dir")
    parser.add_option("--result_file",dest="result_file",default="./result")
    (options, args) = parser.parse_args()

    word_vec_file = options.vector
    word_sim_dir = options.similarity
    reult_file = options.result_file
    # word_sim_dir = "./data/en"
    # word_vec_file = "./filtered.txt"
    outfile = open(reult_file, 'a')
    sys.stdout = outfile

    word_vecs = read_word_vectors(word_vec_file)
    print('=================================================================================')
    print("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
    print('=================================================================================')

    for i, filename in enumerate(os.listdir(word_sim_dir)):
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename),'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vecs and word2 in word_vecs:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
            else:
                not_found += 1
            total_size += 1
        print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size), end=""),
        print("%15s" % str(not_found), end=""),
        print("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
    outfile.close()