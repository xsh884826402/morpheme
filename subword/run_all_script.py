import os
def f1():
    embed_size = 300
    epochs = 40
    test_model_dir = 'models/bilstm_' + str(embed_size) + '/'
    for i in range(epochs):
        cmd = 'python '+'test.py ' + '--embed_size '+str(embed_size) + ' --test_model_path '+\
              test_model_dir+'models_epoch'+str(i)
        print(cmd)
        os.system(cmd)
def f2():
    input_dir_lists =['bilstm_100/', 'bilstm_300/', 'bilstm_500/']
    for input_dir in input_dir_lists:
        input_dir = 'models/'+input_dir
        output_path = input_dir + 'results/'+'summarize.txt'
        with open(output_path, 'w') as f:
            for i in range(40):
                file = input_dir + 'results/' + 'perfomancepred_models_epoch'+str(i)
                with open(file, 'r') as fr:
                    f.write(fr.readline())
def f3():
    input_dir_lists =['bilstm_100/', 'bilstm_300/', 'bilstm_500/']
    for input_dir in input_dir_lists:
        input_dir = 'models/' + input_dir
        output_path = input_dir + 'results/' + 'summarize.txt'
        output_f1_path = input_dir + 'results/' + 'sum_f1score.txt'
        output_pk_path = input_dir + 'results/' + 'sum_pk.txt'
        output_wd_path = input_dir + 'results/' + 'sum_wd.txt'

        with open(output_path, ) as f:
            f1_list = []
            pk_list = []
            wd_list = []
            for line in f:
                line = line.strip().split()
                f1_list.append(line[8])
                pk_list.append(line[10])
                wd_list.append(line[12])
            with open(output_f1_path, 'w') as fw:
                fw.write("\n".join(f1_list))
            with open(output_pk_path, 'w') as fw:
                fw.write("\n".join(pk_list))
            with open(output_wd_path, 'w') as fw:
                fw.write("\n".join(wd_list))

if __name__ == '__main__':
    f3()
