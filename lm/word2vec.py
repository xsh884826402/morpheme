import tensorflow as tf
import os
import numpy as np
import time


def shuffle(infile, outfile, batch_size):
    with open(infile, 'r') as f:
        contents = f.readlines()
        contents = [item.strip() for item in contents]
        shuff_index = list(range(len(contents) // batch_size))
        np.random.shuffle(shuff_index)
        length = len(contents)
    with open(outfile, 'w') as fw:
        for index in shuff_index:
            for line in contents[index * batch_size:(index + 1) * batch_size]:
                fw.write(line + '\n')
    print('Shuffle Successfully')
    return length


def generate_batch(infile, batchsize):
    if batchsize == 0:
        print('batchsize must > 0')
        return None, None
    x_batch, y_batch = [], []
    count = 0
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            x = line[:-1]
            y = line[-1]
            x_batch.append(x)
            y_batch.append([y])
            count += 1
            if count == batchsize:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
                count = 0

def train_epoch(model,
                sess,
                infile,
                model_path,
                word2id_path
                ):

    min_epoch = 0
    # word2id
    word2id = {}
    with open(word2id_path, 'r') as f:
        for line in f:
            word, id = line.strip().split()
            word2id[word] = id

    saver = tf.train.Saver(tf.trainable_variables())

    if not os.path.exists(model_path):
        print('No model path')

        os.mkdir(model_path)
        print('Build the model path')
    ckpt = tf.train.get_checkpoint_state(model_path)


    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        min_epoch = os.path.basename(ckpt.model_checkpoint_path)
        min_epoch = min_epoch.replace('models_epoch', '')
        min_epoch = int(min_epoch)+1
    else:
        sess.run(tf.initialize_all_variables())

    summary_writer = tf.summary.FileWriter("log/", sess.graph)

    # load data
    shuffle_file_path = os.path.join(os.path.dirname(infile), 'shuffle_'+os.path.basename(infile))
    length = shuffle(infile, shuffle_file_path, model.batch_size)
    print('min_epoch {} max_epoch {}'.format(min_epoch,model.max_epochs))
    for epoch in range(min_epoch, model.max_epochs):
        print('%d Epoch starts, Training....' % (epoch))
        start_time = time.time()
        mean_loss = []
        batch_count = 0
        generator = generate_batch(shuffle_file_path, model.batch_size)
        for index in range(length//model.batch_size):
            # generate the data feed dict
            input_batch, label_batch = next(generator)
            # print('In training '*10)
            # print('input batch',np.shape(input_batch))
            # print('label_batch',np.shape(label_batch))
            feed = {
                model.input_placeholder: input_batch,
                model.label_placeholder: label_batch,
            }
            _, loss_step,embedding = sess.run(
                [model.train_op, model.loss, model.embeddings], feed_dict=feed)
            if index ==length//model.batch_size-1:
                embedding_path = os.path.join(model_path, 'embedding')
                if not os.path.exists(embedding_path):
                    os.makedirs(embedding_path)
                embedding_path = os.path.join(embedding_path, 'word_vector_'+str(epoch))

                print('Storing embedding in {}'.format(embedding_path))

                with open(embedding_path, 'w') as f:
                    for word,id in word2id.items():
                        f.write(word + '\t' + ' '.join(list(map(str, embedding[int(id)]))) + '\n')
            loss_step = np.sum(loss_step)
            mean_loss.append(loss_step)
            batch_count += 1
            if batch_count % 1000 == 0:
                print('step %d / %d : time: %ds, loss : %f' %
                      (batch_count, length // model.batch_size,
                       time.time() - start_time, np.mean(mean_loss)))
                mean_loss = []

        train_loss = np.mean(mean_loss)
        mean_loss = []
        print('epoch: {} loss: {}'.format(epoch, train_loss))
        tf.summary.scalar('loss', train_loss)
        tf.summary.merge_all()
        print('epoch_time: %ds' % (time.time() - start_time))
        save_path = saver.save(
            sess, os.path.join(model_path, "models_epoch" + str(epoch)))
class Word2Vec(object):
    def __init__(self, config):

        self.batch_size = config.batch_size
        self.vocabulary_size = config.vocabulary_size
        self.embedding_size = config.embedding_size
        self.skip_window = config.skip_window
        self.max_epochs = config.max_epochs
        self.subword_count = 1


        if config.num_skip:
            self.num_skip = config.num_skip
        else:
            self.num_skip = 2*self.skip_window
        self.num_sampled = config.num_sampled

        # model

        self.input_placeholder = tf.placeholder(tf.int32,)
        self.label_placeholder = tf.placeholder(tf.int32,)
        label = tf.reshape(self.label_placeholder,shape=[self.batch_size,1])
        inputs = tf.reshape(self.input_placeholder, shape=[self.batch_size, self.subword_count])

        self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        inputs = tf.nn.embedding_lookup(self.embeddings, inputs)
        inputs = tf.reshape(inputs, shape=[self.batch_size, self.embedding_size])



        # 使用 NCE Loss 作为训练的优化目标
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size]))
        nce_bias = tf.Variable(tf.zeros([self.vocabulary_size]))

        # 使用 tf.nn.nce_loss 计算学习出的词向量 embed 在训练数据上的 loss，并使用 tf.reduce_mean 进行汇总

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_bias, labels=label, inputs=inputs,
                                       num_sampled=self.num_sampled,
                                       num_classes=self.vocabulary_size))

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)


class Word2Vec_subword_extend(object):
    def __init__(self, config):

        self.batch_size = config.batch_size
        self.embedding_size = config.embedding_size
        self.skip_window = config.skip_window
        self.max_epochs = config.max_epochs
        self.subword_count = 4

        if config.if_extend:
            self.vocabulary_size = config.vocabulary_size + config.vocabulary_size//4
        else:
            self.vocabulary_size = config.vocabulary_size
        if config.num_skip:
            self.num_skip = config.num_skip
        else:
            self.num_skip = 2*self.skip_window
        self.num_sampled = config.num_sampled

        # model

        self.input_placeholder = tf.placeholder(tf.int32,)
        self.label_placeholder = tf.placeholder(tf.int32,)
        label = tf.reshape(self.label_placeholder,shape=[self.batch_size,1])

        self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        inputs_subword = tf.reduce_sum(inputs, axis=1)
        inputs_subword = inputs_subword / self.subword_count

        # 使用 NCE Loss 作为训练的优化目标
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size]))
        nce_bias = tf.Variable(tf.zeros([self.vocabulary_size]))

        # 使用 tf.nn.nce_loss 计算学习出的词向量 embed 在训练数据上的 loss，并使用 tf.reduce_mean 进行汇总

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_bias, labels=label, inputs=inputs_subword,
                                       num_sampled=self.num_sampled,
                                       num_classes=self.vocabulary_size))

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    # def train_epoch(self,
    #                 sess,
    #                 infile,
    #                 model_path,
    #                 word2id_path
    #                 ):
    #     # word2id
    #     word2id = {}
    #     with open(word2id_path, 'r') as f:
    #         for line in f:
    #             word, id = line.strip().split()
    #             word2id[word] = id
    #
    #     saver = tf.train.Saver(tf.trainable_variables())
    #
    #     if not os.path.exists(model_path):
    #         print('No model path')
    #         os.mkdir(model_path)
    #         print('Build the model path')
    #     ckpt = tf.train.get_checkpoint_state(model_path)
    #
    #
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         sess.run(tf.initialize_all_variables())
    #
    #     summary_writer = tf.summary.FileWriter("log/", sess.graph)
    #
    #     # load data
    #     shuffle_file_path = os.path.join(os.path.dirname(infile), 'shuffle_'+os.path.basename(infile))
    #
    #
    #
    #     for epoch in range(self.max_epochs):
    #         length = shuffle(infile, shuffle_file_path, self.batch_size)
    #         print('%d Epoch starts, Training....' % (epoch))
    #         start_time = time.time()
    #         mean_loss = []
    #         batch_count = 0
    #         generator = generate_batch(shuffle_file_path, self.batch_size)
    #         for index in range(length//self.batch_size):
    #             # generate the data feed dict
    #             input_batch, label_batch = next(generator)
    #             feed = {
    #                 self.input_placeholder: input_batch,
    #                 self.label_placeholder: label_batch,
    #             }
    #             _, loss_step,embedding = sess.run(
    #                 [self.train_op, self.loss, self.embeddings], feed_dict=feed)
    #             if index ==length//self.batch_size-1:
    #                 embedding_path = os.path.join(model_path, 'embedding')
    #                 if not os.path.exists(embedding_path):
    #                     os.makedirs(embedding_path)
    #                 embedding_path = os.path.join(embedding_path, 'word_vector_'+str(epoch))
    #
    #                 print('Storing embedding in {}'.format(embedding_path))
    #
    #                 with open(embedding_path, 'w') as f:
    #                     for word,id in word2id.items():
    #                         f.write(word + '\t' + ' '.join(list(map(str, embedding[int(id)]))) + '\n')
    #             loss_step = np.sum(loss_step)
    #             mean_loss.append(loss_step)
    #             batch_count += 1
    #             if batch_count % 1000 == 0:
    #                 print('step %d / %d : time: %ds, loss : %f' %
    #                       (batch_count, length // self.batch_size,
    #                        time.time() - start_time, np.mean(mean_loss)))
    #                 mean_loss = []
    #
    #         train_loss = np.mean(mean_loss)
    #         mean_loss = []
    #         print('epoch: {} loss: {}'.format(epoch, train_loss))
    #         tf.summary.scalar('loss', train_loss)
    #         tf.summary.merge_all()
    #         print('epoch_time: %ds' % (time.time() - start_time))
    #         save_path = saver.save(
    #             sess, os.path.join(model_path, "models_epoch" + str(epoch)))


class Word2Vec_subword_extend_concat(object):
    def __init__(self, config):

        self.batch_size = config.batch_size
        self.embedding_size = config.embedding_size
        self.skip_window = config.skip_window
        self.max_epochs = config.max_epochs
        self.subword_count = 4

        if config.if_extend:
            self.vocabulary_size = config.vocabulary_size + config.vocabulary_size//4
        else:
            self.vocabulary_size = config.vocabulary_size
        if config.num_skip:
            self.num_skip = config.num_skip
        else:
            self.num_skip = 2*self.skip_window
        self.num_sampled = config.num_sampled

        # model

        self.input_placeholder = tf.placeholder(tf.int32,)
        self.label_placeholder = tf.placeholder(tf.int32,)
        label = tf.reshape(self.label_placeholder, shape=[self.batch_size,1])

        self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)

        inputs_subword = tf.reshape(inputs, shape=[self.batch_size, self.subword_count * self.embedding_size])
        # 使用 NCE Loss 作为训练的优化目标
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.subword_count * self.embedding_size]))
        nce_bias = tf.Variable(tf.zeros([self.vocabulary_size]))

        # 使用 tf.nn.nce_loss 计算学习出的词向量 embed 在训练数据上的 loss，并使用 tf.reduce_mean 进行汇总

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_bias, labels=label, inputs=inputs_subword,
                                       num_sampled=self.num_sampled,
                                       num_classes=self.vocabulary_size))

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)





