import os
import sentencepiece as spm
def generate_sp_result(input_file, text8_file ='../data/text8_oneword_perline'):

    sp = spm.SentencePieceProcessor(model_file='m.model')
    with open(input_file, 'w') as fw:
        with open(text8_file, 'r') as fr:
            for line in fr:
                line = line.strip()
                res = sp.encode(line, out_type=str)
                # print(res[0], res[0][0], ord(res[0][0]))
                if res[0][0]==chr(9601):
                    res[0] = res[0][1:]
                fw.write(" ".join(res)+'\n')
    print('generate text8_sp_result successfully')

if __name__ == '__main__':

    input_file1 = '../data/text8_oneword_perline'
    input_file2 = 'text8_sp_result'
    if not os.path.exists(input_file2):
        generate_sp_result(input_file2)
    output_path = '../data/sp_subword.txt'
    with open(input_file1) as f:
        inputs1 = f.readlines()
    with open(input_file2) as f:
        inputs2 = f.readlines()
    with open(output_path, 'w') as fw:

        for word,subword in zip(inputs1, inputs2):
            word = word.strip()
            subwords = subword.strip().split()
            if len(subwords)==1:
                if subwords[0]==word:
                    subwords = ['#', '#', '#']
                else:
                    print('Error word:{} subword:{}'.format(word," ".join(subwords)))
                    continue
            elif len(subwords)==2:
                subwords.extend(['#'])

            elif len(subwords)>3:
                subwords = [subwords[0]] + ["".join(subwords[1:-1])]+ [subwords[-1]]

            assert len(subwords)==3
            print('subword',subwords)
            fw.write(word+'\t'+" ".join(subwords)+'\n')
    print('Generating Successfully')



