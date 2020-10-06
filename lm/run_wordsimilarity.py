import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dir')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()
    root_path = '/home/ubuntu/User/xsh/Paper/'

    vector_directory_path =args.vector_dir+'word_vector_'
    similarity_py_path =root_path + 'en_embedding_similarity/all_wordsim.py'
    similarity_data_path = root_path + 'en_embedding_similarity/data/en'
    result_path = root_path+'lm/'+ args.output_path
    max_epoch = 60
    for file_index in range(max_epoch):

        # if file_index%2 != 0:
        #     continue

        new_cmd = 'python '+ similarity_py_path + ' --vector=' +vector_directory_path+str(file_index)+' --similarity='+ similarity_data_path +' --result_file='+result_path
        os.system(new_cmd)