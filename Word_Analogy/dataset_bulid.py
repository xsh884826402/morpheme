import re
import collections
def filter_words(input_path, output_path):
    with open(input_path, 'r') as fr:
        words = []
        for line in fr:
            words.extend(line.strip().split())
    print('Reading  from {} Successfully'.format(input_path))
    words = collections.Counter(words)
    print("the number of words is {}".format(len(words)))
    with open(output_path, 'w') as fw:
        for word in words.keys():
            fw.write(word+" "+str(words[word])+'\n')

    print('Writing words to {} successfully'.format(output_path))


def rule1(word_count_dict, vocabulary, result_path,rule,threshold=100):
    p = re.compile(rule)
    strings = []
    for word in vocabulary:
        if p.search(word):
            new_word = p.sub("", word)
            if new_word in word_count_dict:
                strings.append([new_word+' '+word, word_count_dict[word]])
    strings = sorted(strings, key= lambda item :item[1], reverse=True)
    print(strings[:100])
    strings = [item[0] for item in strings[:threshold]]
    with open(result_path, 'a') as fw:
        for string in strings:
            fw.write(string+'\n')

def build_result(input_path, output_path):
    with open(input_path, 'r') as fr:
        with open(output_path, 'a') as fw:
            pairs = []
            for line in fr:
                line = line.strip()
                if line.startswith(':'):
                    if not pairs:
                        fw.write(line+'\n')
                    else:
                        to_write = []
                        for i,pair in enumerate(pairs):
                            for j in range(i+1,len(pairs)):
                                to_write.append(pair+' '+pairs[j])
                        fw.write("\n".join(to_write)+'\n')
                        fw.write(line+'\n')
                        pairs = []

                else:
                    pairs.append(line)
            to_write = []
            for i, pair in enumerate(pairs):
                for j in range(i + 1, len(pairs)):
                    to_write.append(pair + ' ' + pairs[j])
            fw.write("\n".join(to_write))

    print("build results successfully")






if __name__ == '__main__':
    input_path = '../data/text8'
    output_path = './Data/text8_words'
    temp_result_path = './Data/temp_morpheme_analogy.txt'
    result_path = './Data/morpheme_analogy.txt'
    # filter_words(input_path, output_path)

    word_count_dict = {}
    with open(output_path) as fr:
        for line in fr:
            key,value = line.strip().split()
            word_count_dict[key] = value
    #vocabulary
    with open('../data/Visual_vector/raw_word_vector_8', 'r') as fr:
        vocabulary = []
        for line in fr:
            line = line.strip()
            vocabulary.append(line.split()[0])
    vocabulary = set(vocabulary)

    #rule1(word_count_dict, vocabulary, temp_result_path, 'ity$', threshold=300)
    build_result(temp_result_path, result_path)
