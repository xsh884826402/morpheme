# coding: utf-8
import os
import re
import sys
import codecs
import random
from opencc import OpenCC

word2id = {}


def make_dict(vocab_file):
    with codecs.open(vocab_file, 'r') as fr:
        cnt = 0
        for item in fr:
            item = item.strip().split()
            word2id[item[0]] = item[1]


class TextProcess(object):
    def __init__(self):
        self.p_traditional2simple = OpenCC('t2s')
        super().__init__()

    def traditional2simple(self, strs):
        return self.p_traditional2simple.convert(strs)

    def is_english(self, strs):
        regex = '[a-zA-Z]+'
        return re.match(regex, strs)

    def is_num(self, strs):
        regex = '[0-9]+.*[a-zA-Z].*'
        return re.match(regex, strs)

    def contain_english(self, strs):
        regex = '.*[a-zA-Z].*'
        return re.match(regex, strs)

    def is_chinese(self, strs):
        regex = '[\u4e00-\u9fa5]'
        return re.match(regex, strs)

    def char_chinese(self, char):
        return '\u4e00' <= char <= '\u9fff'


def wiki_data_process(infile):
    if not os.path.exists(infile):
        print('The file provided {} is not found!'.format(infile))
        sys.exit()
    label = -1
    handler = TextProcess()

    def check_valid(line):
        if handler.is_num(line) or handler.is_english(line):
            return False
        return True

    with codecs.open(infile, 'r', encoding='utf-8') as fr:
        content = ''
        _all = []
        for line in fr:
            if not check_valid(line):
                continue
            else:
                line = handler.traditional2simple(line)
                if line.startswith('=='):
                    if content != '':
                        _all.append(content)
                    label = 1
                    content = ''
                else:
                    new_line = ''
                    for item in ' '.join(line).split():
                        if handler.char_chinese(item):
                            try:
                                ids = word2id[item]
                            except:
                                ids = word2id['<unk>']
                            new_line += ids + ' '
                    if new_line and label != -1:
                        content += new_line + '\t' + str(label) + '\n'
                        label = 0
            if len(_all) % 1000 == 0:
                print('====== process {} lines ==========='.format(len(_all)))
        _all.append(content)
    return _all


def train_dev_split(infile, train_file, dev_file, vocab_file, ratio=0.9):
    if os.path.exists(train_file) and os.path.exists(dev_file):
        return
    if not os.path.exists(infile):
        print('The {} is not found!'.format(infile))
        sys.exit()
    make_dict(vocab_file)
    content_lst = wiki_data_process(infile)
    random.shuffle(content_lst)
    le = int(len(content_lst) * ratio)
    cnt = 0
    with codecs.open(train_file, 'w') as fw1, codecs.open(dev_file,
                                                          'w') as fw2:
        for line in content_lst:
            if cnt < le:
                fw1.write(line)
            else:
                fw2.write(line)
            cnt += 1


def load_data(train_file):
    with open(train_file, 'r') as f:
        x_all = []
        y_all = []
        for line in f:
            line = line.strip().split('\t')
            x_list = line[0].strip().split(' ')
            y_label = line[1]
            for i in range(0, len(x_list)):
                x_list[i] = int(x_list[i])
            y_label = int(y_label)
            x_all.append(x_list)
            y_all.append(y_label)
    return x_all, y_all


def write_data(lists, result_file):
    par_dir = os.path.dirname(result_file)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    with open(result_file, 'w') as f:
        for line in lists:
            f.write(' '.join([str(item) for item in line]) + '\n')


def get_refs(true_file):
    refs = []
    with open(true_file, 'r') as fr2:
        for line in fr2:
            line = line.split()
            refs.append(int(line[-1]))
    return refs


def get_preds(pred_file, threshold):
    preds = []
    with open(pred_file, 'r') as fr1:
        for line in fr1:
            line = line.split()
            cnt = 0
            for item in line:
                if item == '1':
                    cnt += 1
            label = 1 if cnt > threshold else 0
            preds.append(label)
    return preds


# def get_metrics(preds, refs, threshold, model_name='best'):
#     assert len(preds) == len(refs)
#
#     one_positive = 0
#     one_negative = 0
#     zero_positive = 0
#     zero_negative = 0
#     for pred, true in zip(preds, refs):
#         if pred != true:
#             if true == 1:
#                 one_negative += 1
#                 #pred = 0
#             else:
#                 zero_negative += 1
#                 # pred = 1
#         else:
#             if true == 1:
#                 one_positive += 1
#                 # pred = 1
#             else:
#                 zero_positive += 1
#                 # pred = 0
#
#     precision = 1.0 * one_positive / (zero_negative + one_positive)
#     recall = 1.0 * one_positive / (one_negative + one_positive)
#     F1 = 2 * precision * recall / (precision + recall)
#     with open('results/performance.txt', 'a') as fw:
#         fw.write(
#             'model_name: {} threshold: {} precision: {} recall: {} F1: {}\n'.
#             format(model_name, threshold, precision, recall, F1))
#     return precision, recall, F1

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


def get_metrics(preds,refs, path,model_name='bests',k=20):
    assert  len(preds)==len(refs)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for pre,ref in zip(preds,refs):
        if pre ==1:
            if ref==1:
                TP += 1
            else:
                FP += 1
        else:
            if ref==0:
                TN += 1
            else:
                FN += 1
    precision = 1.0*TP/(TP+FP)
    recall = 1.0*TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    pk_score=pk(list(refs), list(preds), k=k)
    windiff_score = windowdiff(list(refs), list(preds), k=k)

    file_path = os.path.dirname(path)+'/perfomance'+os.path.basename(path)
    with open(file_path,'w') as fw:
        fw.write(
            'model_name: {} precision: {} recall: {} F1_score : {} pk: {} windiff: {}\n'.format(model_name, precision, recall, F1, pk_score, windiff_score)
        )



#
# def eval_metrics(preds_file, model_name='best'):
#     thresholds = [20, 21, 22, 23, 24]
#     refs = get_refs('data/dev.txt')
#
#     for threshold in thresholds:
#         preds = get_preds(preds_file, threshold)
#         get_metrics(preds, refs, threshold, model_name)




# modified by xushenghua
def eval_metrics(preds_file, model_name='best'):
    refs = get_refs('./data/dev.txt')

    preds = get_refs(preds_file)
    get_metrics(preds, refs, preds_file, model_name)

if __name__ == '__main__':
    eval_metrics('./data/pred.models_epoch19')
