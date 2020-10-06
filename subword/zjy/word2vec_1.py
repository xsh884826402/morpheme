# coding: utf-8
from __future__ import division
import collections
import math
# import os
import random
import zipfile
import numpy as np
import tensorflow as tf
import Trie
#求统计前缀后缀  存进字典中
with open("inputdata/pre.txt", 'r') as f:
    words = f.read().splitlines()
flag = 50000
dictionary2 = dict()
for i in words:
    dictionary2[i] = flag
    flag = flag + 1
with open("inputdata/suf.txt", 'r') as f:
    words = f.read().splitlines()
dictionary3 = dict()
for i in words:
    dictionary3[i] = flag
    flag = flag + 1
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
filename = "test/text8.zip"
words = read_data(filename)
print ('Data size', len(words))

# 将词存入 word 列表中
vocabulary_size1 = 53682
vocabulary_size = 50000  # 将出现频率最高的 50000 个单词放入 count 列表中，然后放入 dicionary 中

def build_dataset(words):
    count = [['UNK', -1]]  # 前面是词汇，后面是出现的次数，这里的 -1 在下面会填上 UNK 出现的频数
    # 将出现频次最高的 50000 个词存入count
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # -1 因为 UNK 已经占了一个了

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # 等价于，就是按 count 中词出现的顺序，分别给他们编号：0 1 2 ...
    #   for i in vocabulary_size:
    #        dictionary[count[i][0]]=i

    # 编码：如果不出现在 dictionary 中，就以 0 作为编号，否则以 dictionary 中的编号编号
    # 也就是将 words 中的所有词的编号存在 data 中，顺带查一下 UNK 有多少个，以便替换 count 中的 -1
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count

    # 编号：词
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # 删除原始单词表，节约内存
y = Trie.Trie()
with open("inputdata/suf.txt",'r') as f:
    content = f.read().splitlines()
for i in content:
    l = list(reversed(i))
    s = ''.join(l)
    y.insert(s)
x = Trie.Trie()
with open("inputdata/pre.txt",'r') as f:
    content = f.read().splitlines()
for i in content:
    x.insert(i)
count1 = []
tem = []
f1 = open("result/ceshi.txt", "a+", encoding='utf-8')
for kk in count:
    i = kk[0]
    a = x.startsWith(i)
    if a != 'None':
        tem = [dictionary2[a]]
    else:
        tem = [dictionary[i]]
    index1 = x.z + 1
    l = list(reversed(i))
    s = ''.join(l)
    b = y.startsWith(s)
    index2 = len(i) - y.z - 1
    if index1 < index2:
        c = i[index1:index2]
    else:
        c = "None"
    if b == "None":
        s = b
        tem.append(dictionary[i])
    else:
        l = list(reversed(b))
        s = ''.join(l)
        tem.append(dictionary3[s])
    tem.append(dictionary[i])
    count1.append(tem)
    f1.write(i + "  " + a + "   " + c + "  " + s + '\n')  # 将每一行打印进test1.txt文件并换行
f1.close()  # 关闭test1.txt文件**


# 生成 Word2Vec 训练样本
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index  # 设为global 因为会反复 generate
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # 将 batch 和 labels 初始化为数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # 对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
    span = 2 * skip_window + 1

    # 创建最大容量为 span 的 deque（双向队列）
    # 在用 append 对 deque 添加变量时，只会保留最后插入的 span 个变量
    buffer = collections.deque(maxlen=span)

    # 从 data_index 开始，把 span 个单词顺序读入 buffer 作为初始值，buffer 中存的是词的编号
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # buffer 容量是 span，所以此时 buffer 已经填满，后续的数据将替换掉前面的数据

    # 每次循环内对一个目标单词生成样本，前方已经断言能整除，这里使用 // 是为了保证结果是 int
    for i in range(batch_size // num_skips):  # //除法只保留结果整数部分（python3中），python2中直接 /
        # 现在 buffer 中是目标单词和所有相关单词
        target = skip_window  # buffer 中第 skip_window 个单词为目标单词（注意第一个目标单词是 buffer[skip_window]，并不是 buffer[0]）
        targets_to_avoid = [skip_window]  # 接下来生成相关（上下文语境）单词，应将目标单词拒绝

        # 每次循环对一个语境单词生成样本
        for j in range(num_skips):
            # 先产生一个随机数，直到随机数不在 targets_to_avoid 中，就可以将之作为语境单词
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)  # 因为这个语境单词被使用了，所以要加入到 targets_to_avoid

            batch[i * num_skips + j] = buffer[skip_window]  # feature 是目标词汇
            labels[i * num_skips + j, 0] = buffer[target]  # label 是 buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# 训练需要的参数
batch_size = 128
embedding_size = 128  # 将单词转为稠密向量的维度，一般是500~1000这个范围内的值，这里设为128
skip_window = 1  # 单词间最远可以联系到的距离
num_skips = 2  # 对每个目标单词提取的样本数

# 生成验证数据，随机抽取一些频数最高的单词，看向量空间上跟它们距离最近的单词是否相关性比较高
valid_size = 16  # 抽取的验证单词数
valid_window = 100  # 验证单词只从频数最高的 100 个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 随机抽取
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量


graph = tf.Graph()
with graph.as_default():
    # 建立输入占位符
    train_inputs = tf.placeholder(tf.int32, shape=[None,None])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  # 将前面随机产生的 valid_examples 转为 TensorFlow 中的 constant

    with tf.device('/gpu:0'):  # 限定所有计算在 CPU 上执行
        # 随机生成所有单词的词向量 embeddings，单词表大小 5000，向量维度 128
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size1, embedding_size], -1.0, 1.0))

        # 查找 train_inputs 对应的向量 embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        embed2 = tf.reduce_sum(embed, axis=1)
        embed1 = embed2 / 3
        # 使用 NCE Loss 作为训练的优化目标
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size1, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_bias = tf.Variable(tf.zeros([vocabulary_size1]))

    # 使用 tf.nn.nce_loss 计算学习出的词向量 embed 在训练数据上的 loss，并使用 tf.reduce_mean 进行汇总
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_bias, labels=train_labels, inputs=embed1,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size1))
    '''loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels, inputs=embed1, num_sampled=num_sampled,
                       num_classes=vocabulary_size1))
                       '''
    # 定义优化器为 SGD，且学习速率为 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 计算嵌入向量 embeddings 的 L2 范数 norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 标准化
    normalized_embeddings = embeddings / norm
    # 查询验证单词的嵌入向量，并计算验证单词的嵌入向量与词汇表中所有单词的相似性
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 初始化所有模型参数
    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    average_loss = 0
    f1 = open("result/output1.txt", "a+", encoding='utf-8')
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        batch = []
        for i in range(128):
            batch.append(count1[batch_inputs[i]])
        feed_dict = {train_inputs: batch, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            log_str = 'Average loss at step {} : {}'.format(step, average_loss)
            f1.write(log_str + '\n')  # 将每一行打印进test1.txt文件并换行
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {} :'.format(valid_word)

                for k in range(top_k):
                    if nearest[k] < 50000:
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '{} {},'.format(log_str, close_word)
                f1.write(log_str + '\n')  # 将每一行打印进test1.txt文件并换行
        final_embeddings = normalized_embeddings.eval()
    f1.close()  # 关闭test1.txt文件**
