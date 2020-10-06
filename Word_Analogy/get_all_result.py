def write_result(output_path):
    vectors = ['raw', 'bpe', 'infer',]
    s_path_1 ='_word_vector_8_top'
    s_path_2 = '_result.txt'
    with open(output_path, 'w') as fw:
        for vector in vectors:
            fw.write(vector+'\n')
            for i in range(2,50):
                if i%2==0:
                    with open(vector+s_path_1+str(i)+s_path_2, 'r') as fr:
                        for line in fr:
                            line = line.strip()
                            if line.startswith('Total acc'):
                                new_line = line.split(':')[1]
                                fw.write(new_line+'\n')
    print('Write all result in {}'.format(output_path))

if __name__ == '__main__':
    output_path = 'Data/all_result.txt'
    write_result(output_path)
