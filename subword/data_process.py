# coding: utf-8
from Trie import Trie
#from inference import inference_main
import os
import random
sql_len = 20

def word_to_subword(infile, outfile):
    preTrie = Trie()
    with open('../data/pre.txt', 'r') as f:
        for line in f:
            line = line.strip()
            preTrie.insert(line)

    suffTrie = Trie()
    with open('../data/suff.txt', 'r') as f:
        for line in f:
            line = line.strip()
            suffTrie.insert(line[::-1])
    print(preTrie.startsWith('including'))
    with open(outfile, 'w') as fw:
        with open(infile, 'r') as fr:
            for line in fr:
                for word in line.strip().split():
                    new_line = []
                    prefix = preTrie.startsWith(word)
                    sufffix = suffTrie.startsWith(word[::-1])
                    if sufffix: sufffix = sufffix[::-1]
                    if prefix:
                        if sufffix:
                            if len(prefix)+len(sufffix)>=len(word):
                                new_line.extend([prefix, word[len(prefix):], ''])
                            else:
                                new_line.extend([prefix, word[len(prefix):-len(sufffix)], sufffix])
                        else:
                            new_line.extend([prefix,word[len(prefix):],''])

                    else:
                        if sufffix:
                            new_line.extend(['',word[:-len(sufffix)],sufffix])
                        else:
                            new_line.extend(['','',''])
                    new_line = [item  if item!='' else '#' for item in new_line]
                    fw.write(word+'\t'+' '.join(new_line)+'\n')
    print('word to subword Successfully')


def word_to_subword_using_model(infile, outfile):
    pass
def make_label(infile, outfile):
    with open(infile, 'r') as f:
        with open(outfile, 'w') as fw:
            for line in f:
                line = line.strip()
                word, subwords = line.split('\t')
                subwords =subwords.split(' ')
                label = ['0']*len(word)
                index = 0
                for subword in subwords:
                    if subword=='#':
                        continue
                    else:
                        index = index+len(subword)
                        # print('line index:{}'.format(index)+line)
                        if index!=len(word):
                            label[index] = '1'
                fw.write(word+' '+''.join(label)+'\n')
    print('Make label Successfully')

def train_dev_split(infile, train_file, dev_file, vocab_file, ratio=0.8):
    '''
    # shuffle数据并分割训练数据到train_file, dev_file
    :param infile:
    :param train_file:
    :param dev_file:
    :param vocab_file:
    :param ratio:
    :return:
    '''
    if os.path.exists(train_file) and os.path.exists(dev_file):
        print('Train and Dev data exists')
        return
    if not os.path.exists(infile):
        print('The {} is not found!'.format(infile))
        raise Exception
    with open(infile, 'r') as f:
        contents = f.readlines()
    random.shuffle(contents)
    lenn = int(len(contents) * ratio)
    cnt = 0
    with open(train_file, 'w') as fw1, open(dev_file, 'w') as fw2:
        for line in contents:
            if cnt < lenn:
                fw1.write(line)
            else:
                fw2.write(line)
            cnt += 1


def load_data(train_file):
    '''
    # 获取数据,并将字符转换为数字
    :param train_file:
    :return: 二维列表[[int]]
    '''
    with open(train_file, 'r') as f:
        x_all = []
        y_all = []
        for line in f:
            x, y = line.split()
            x = [ord(item)-ord('a') for item in x]
            y = [int(item) for item in y]
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all

def prepare_modelinputs(contents, word2id_path =None,y_flag=True):
    '''
    处理内容，并将内容转换成id
    :param contents: 一维列表，内容是字符串
    :return:
    '''
    if y_flag:
        xs = []
        ys = []
        for content in contents:
            content = content.strip()
            content = content.split()
            x = content[0]
            y = content[-1]
            x = [ord(item)-ord('a') for item in x ]
            y = [int(item) for item in y]
            xs.append(x)
            ys.append(y)
        return xs,ys

    else:
        xs = []
        for content in contents:
            x = [ord(item)-ord('a') for item in content]
            xs.append(x)
        return xs


def prepare_and_write_modelinputs(infile, outfile, word2id_path = None,y_flag =True, encoding='utf-8'):
    contents = []
    with open(infile, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            contents.append(line)
    if y_flag:
        xs, ys = prepare_modelinputs(contents)
        assert len(xs)==len(ys)
        with open(outfile, 'w') as f:
            for x, y in zip(xs, ys):
                f.write(" ".join(list(map(str, x)))+" "+" ".join(list(map(str, y)))+ '\n')
        print('Writing Successfully')




def write_preds(y, outfile):
    '''
    dump the preds to outfile
    :param y:
    :param outfile:
    :return:
    '''
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    with open(outfile, 'w') as fw:
        for line in y:
            line = list(map(str,line))
            fw.write("".join(line)+'\n')
    print('Write preds Successfully!')



def get_refs(infile):
    '''

    :param infile:
    :return: [str]
    '''
    res = []
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()[1]
            res.append(line)
    return res

def get_preds(infile):
    '''

    :param infile:
    :return: [[str]]
    '''
    res = []
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip()
            res.append(line)
    return res

def pk(ref, hyp, k=None, boundary="1"):
    if not isinstance(ref,str):
        ref = "".join(map(str,ref))
    if not isinstance(hyp,str):
        hyp = "".join(map(str,hyp))
    if k is None:
        k = int(round(len(ref) / (ref.count(boundary) * 2.0)))

    err = 0
    for i in range(len(ref) - k + 1):
        r = ref[i : i + k].count(boundary) > 0
        h = hyp[i : i + k].count(boundary) > 0
        if r != h:
            err += 1
    return err / (len(ref) - k + 1.0)


def windowdiff(seg1, seg2, k, boundary="1", weighted=False):
    if not isinstance(seg1,str):
        seg1 = "".join(map(str,seg1))
    if not isinstance(seg2,str):
        seg2 = "".join(map(str,seg2))
    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    if k > len(seg1):
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0
    for i in range(len(seg1) - k + 1):
        ndiff = abs(seg1[i : i + k].count(boundary) - seg2[i : i + k].count(boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (len(seg1) - k + 1.0)


def get_metrics(preds,refs, path, model_name='bests',k=20):
    assert  len(preds)==len(refs)
    preds = "".join(preds)
    refs = "".join(refs)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for pre,ref in zip(preds, refs):
        if pre =='1':
            if ref=='1':
                TP += 1
            else:
                FP += 1
        else:
            if ref=='0':
                TN += 1
            else:
                FN += 1
    precision = 1.0*TP/(TP+FP)
    recall = 1.0*TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)

    pk_score=pk(refs, preds, k=k)
    windiff_score = windowdiff(refs, preds, k=k)

    file_path = os.path.dirname(path)+'/perfomance'+os.path.basename(path)
    with open(file_path, 'w') as fw:
        fw.write(
            'model_name: {} precision: {} recall: {} F1_score : {} pk: {} windiff: {}\n'.format(model_name, precision, recall, F1, pk_score, windiff_score)
        )


def eval_metrics(preds_file, modelname='bests',refs_path ='../data/dev.txt', k=10 ):
    '''

    :param preds_file:the preds_file
    :param modelname: index the model used
    :return:
    '''
    refs = get_refs(refs_path)

    preds = get_preds(preds_file)

    assert len(refs)==len(preds)

    # truncte
    for i in range(len(refs)):
        lenn = len(refs[i])
        if lenn>sql_len:
            refs[i] = refs[i][:sql_len]
            lenn = sql_len
        preds[i] = preds[i][:lenn]
    # print('type refs,preds', type(refs), type(preds))
    get_metrics(preds, refs, preds_file, modelname,k=k)


if __name__ =="__main__" :
    infile = './data/text8'
    word_to_subword(infile, outfile='./data/subword.txt')
    # make_label('./data/subword.txt', './data/input_word_label.txt')
    eval_metrics('./models/bilstm/result/pred_models_epoch38', './models/bilstm/modes_epoch38')