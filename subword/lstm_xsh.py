#coding: utf-8

from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
from keras_preprocessing import sequence
import numpy as np
from data_process import write_preds, eval_metrics
# from data_process import load_data, write_data, eval_metrics
# from eval import windiff_and_pk_metric_ONE_SEQUENCE, precision, recall



def get_maxcnt_val(two_d_arr):
    ans = []
    for item in two_d_arr:
        ans.append(np.argmax(np.bincount(item)))
    return ans
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
    '''

    :param infile: x和y用空格分离 最后一个是y
    :param batchsize:
    :return:
    '''
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

def generate_batch_new(infile, batchsize):
    '''

    :param infile: x和y用空格分离 最后一个是y
    :param batchsize:
    :return:
    '''
    if batchsize == 0:
        print('batchsize must > 0')
        return None, None
    x_batch, y_batch = [], []
    count = 0
    with open(infile, 'r') as f:
        for line in f:
            line = line.strip().split()
            lenn = len(line)//2
            x = line[:lenn]
            y = line[lenn:]
            x_batch.append(x)
            y_batch.append(y)
            count += 1
            if count == batchsize:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
                count = 0


def padding_data(x, maxlen):
    x = sequence.pad_sequences(x,
                               maxlen=maxlen,
                               padding='post',
                               truncating='post')
    return x
def train_epoch(model,
                sess,
                infile,
                model_path,
                # word2id_path='../data/word2id_dict_50000_winsize2'
                ):

    min_epoch = 0
    # word2id
    # word2id = {}
    # with open(word2id_path, 'r') as f:
    #     for line in f:
    #         word, id = line.strip().split()
    #         word2id[word] = id

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    if not os.path.exists(model_path):
        print('No model path')
        os.makedirs(model_path)
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

    for epoch in range(min_epoch, model.max_epochs):
        print('%d Epoch starts, Training....' % (epoch))
        start_time = time.time()
        mean_loss = []
        batch_count = 0
        generator = generate_batch_new(shuffle_file_path, model.batch_size)
        for index in range(length//model.batch_size):
            # generate the data feed dict
            input_batch, label_batch = next(generator)
            seq_len = list(map(len, input_batch))
            input_batch = padding_data(input_batch, model.max_seqlen)
            label_batch = padding_data(label_batch,model.max_seqlen)
            # print('type',[0],type(input_batch[0]),type(input_batch[0][0]))
            feed = {
                model.input_placeholder: input_batch,
                model.label_placeholder: label_batch,
                model.sequence_length_placeholder:seq_len,
            }
            _, loss_step= sess.run(
                [model.train_op, model.loss], feed_dict=feed)
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


def do_evaluation(model, sess, x, y, x_len,):
    x = padding_data(x, model.max_seqlen)
    y = padding_data(y, model.max_seqlen)
    num_steps = len(x) // model.batch_size
    # init_state = sess.run([self.initial_state])
    x_test = []
    y_test = []
    y_refs = []
    mean_loss = []
    for step in list(range(num_steps)):
        input_batch = x[step * model.batch_size:(step + 1) *
                        model.batch_size]
        label_batch = y[step * model.batch_size:(step + 1) *
                        model.batch_size]
        seq_len = x_len[step * model.batch_size:(step + 1) *
                        model.batch_size]
        feed = {
            model.input_placeholder: input_batch,
            model.label_placeholder: label_batch,
            model.sequence_length_placeholder: seq_len
        }

        pred, loss_step = sess.run(
            [model.y_pred, model.loss], feed)
        print('pred shape: ', pred.shape)

        y_test = y_test + pred.tolist()
        x_test = x_test + input_batch.tolist()
        y_refs = y_refs + label_batch.tolist()
        loss_step = np.mean(loss_step)
        mean_loss.append(loss_step)

    left = len(x) % model.batch_size
    input_batch = x[-model.batch_size:]
    label_batch = y[-model.batch_size:]
    seq_len = x_len[-model.batch_size:]
    feed = {
        model.input_placeholder: input_batch,
        model.label_placeholder: label_batch,
        model.sequence_length_placeholder: seq_len
    }

    pred = sess.run(model.y_pred, feed)
    y_test = y_test + pred.tolist()[-left:]
    x_test = x_test + input_batch.tolist()[-left:]
    y_refs = y_refs + label_batch.tolist()[-left:]
    print('in do evaluation, type y,y_test',type(y_test),y_test[1])

    return y_test

def test(
         model,
         sess,
         x_test,
         y_test,
         x_len_test,
         model_path,
         ):
    write_file = os.path.join(os.path.dirname(model_path), 'results/'+'pred_'+os.path.basename(model_path))
    print('Start do eval...')
    if not os.path.exists(write_file):
        ytest = do_evaluation(model, sess, x_test, y_test,
                                          x_len_test)
        write_preds(ytest, write_file)
    eval_metrics(write_file)

def infer(
        model,
        sess,
        x_infer,
        x_len_infer,
):
    print('in infering\n')
    # print('x,',len(x_infer[0]), x_infer[0])
    fake_y = x_infer
    y_pred = do_evaluation(model, sess, x_infer, fake_y,x_len_infer)
    return y_pred

class LSTM_Dynamic(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.max_epochs = config.max_epochs
        self.label_kinds = config.label_kinds
        self.if_train = config.if_train
        self.dev_file = config.dev_file
        self.is_biLSTM = config.is_biLSTM
        self.max_seqlen = config.max_seqlen

        self.dropout_rate = 0.5

        # self.initial_state_placeholder = tf.placeholder(tf.float32)
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=[self.batch_size, None])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                )
        # self.label_placeholder = tf.reshape(self.label_placeholder, shape=[self.batch_size,])
        self.sequence_length_placeholder = tf.placeholder(tf.int32)

        self.embed = tf.get_variable(name="Embedding",
                                     shape=[self.vocab_size, self.embed_size])
        #modified by shenghua
        inputs = tf.nn.embedding_lookup(self.embed, self.input_placeholder)

        ##initial state
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                               forget_bias=0.0)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                               forget_bias=0.0)
        if self.if_train:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell_fw, output_keep_prob=(1 - self.dropout_rate))
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell_bw, output_keep_prob=(1 - self.dropout_rate))

        self.initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        self.initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            sequence_length=self.sequence_length_placeholder)


        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_seqlen, self.hidden_size*2])

        W = tf.get_variable('Weights',
                            shape=[self.hidden_size*2, self.label_kinds])

        b = tf.get_variable('Bias', shape=[self.label_kinds])

        #y_pred shape: (batch_size*time_steps, label_kinds)
        y_pred = tf.matmul(outputs, W) + b

        self.y_pred = tf.argmax(y_pred, 2)
        self.labels = tf.reshape(
            tf.one_hot(self.label_placeholder,
                       self.label_kinds,
                       dtype=tf.int32), [self.batch_size, self.max_seqlen, self.label_kinds])

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=self.labels,)
        self.train_op = tf.train.AdamOptimizer(0.0005).minimize(self.loss)

    def do_evaluation(self, sess, X, y, X_len, write_file):
        (X, y) = self.padding_data(X, y, self.max_seqlen)
        num_steps = len(X) // self.batch_size
        # init_state = sess.run([self.initial_state])
        x_test = []
        y_test = []
        y_refs = []
        mean_loss = []
        for step in list(range(num_steps)):
            input_batch = X[step * self.batch_size:(step + 1) *
                            self.batch_size]
            label_batch = y[step * self.batch_size:(step + 1) *
                            self.batch_size]
            seq_len = X_len[step * self.batch_size:(step + 1) *
                            self.batch_size]
            feed = {
                self.input_placeholder: input_batch,
                self.label_placeholder: label_batch,
                self.sequence_length_placeholder: seq_len
            }

            pred, loss_step = sess.run(
                [self.y_pred, self.loss], feed)
            # print('pred shape: ', pred.shape)

            y_test = y_test + pred.tolist()
            x_test = x_test + input_batch.tolist()
            y_refs = y_refs + label_batch.tolist()
            loss_step = np.mean(loss_step)
            mean_loss.append(loss_step)

        left = len(X) % self.batch_size
        input_batch = X[-self.batch_size:]
        label_batch = y[-self.batch_size:]
        seq_len = X_len[-self.batch_size:]
        feed = {
            self.input_placeholder: input_batch,
            self.label_placeholder: label_batch,
            self.sequence_length_placeholder: seq_len
        }

        pred = sess.run(self.y_pred, feed)
        y_test = y_test + pred.tolist()[-left:]
        x_test = x_test + input_batch.tolist()[-left:]
        y_refs = y_refs + label_batch.tolist()[-left:]
        print('in do evaluation',type(y_test),y_test[1])
        write_preds(y_test, write_file)

        return x_test, y_test

    # def padding_data(self, x, y, maxlen):
    #     x = np.array(x)
    #     x = sequence.pad_sequences(x,
    #                                maxlen=maxlen,
    #                                padding='post',
    #                                truncating='post')
    #
    #     y = sequence.pad_sequences(y,
    #                                maxlen = maxlen,
    #                                padding='post',
    #                                truncating='post')
    #     return [x, y]

    def test(self,
             sess,
             model,
             X_test,
             y_test,
             X_len_test,
             dict_file="./data/char.dict",
             resultFile="result.txt"):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, model)  #load model
        write_file = resultFile+'pred_' + os.path.basename(model)
        print('Start do eval...')
        if not os.path.exists(write_file):
            xtest, ytest = self.do_evaluation(sess, X_test, y_test,
                                              list(X_len_test), write_file)
        eval_metrics(write_file, modelname=model)


    def train_epoch(self,
                    sess,
                    train_file,
                    X_train,
                    y_train,
                    X_len,
                    model_path,
                    restore_model=False):


        saver = tf.train.Saver(tf.trainable_variables())
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        ckpt = tf.train.get_checkpoint_state(model_path)


        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        summary_writer = tf.summary.FileWriter("log/", sess.graph)

        for epoch in range(self.max_epochs):
            print('%d Epoch starts, Training....' % (epoch))
            start_time = time.time()
            mean_loss = []
            # state = sess.run([self.initial_state])

            shuf_step = list(range(len(X_train) // self.batch_size))
            np.random.shuffle(shuf_step)
            batch_count = 0
            for step in shuf_step:
                # generate the data feed dict
                input_batch = X_train[step * self.batch_size:(step + 1) *
                                      self.batch_size]
                label_batch = y_train[step * self.batch_size:(step + 1) *
                                      self.batch_size]

                seq_len = X_len[step * self.batch_size:(step + 1) *
                                self.batch_size]

                (input_batch,
                 label_batch) = self.padding_data(input_batch,
                                               label_batch,
                                               self.max_seqlen
                                               )
                seq_len = [min(x, self.max_seqlen) for x in seq_len]

                feed = {
                    self.input_placeholder: input_batch,
                    self.label_placeholder: label_batch,
                    self.sequence_length_placeholder: seq_len
                }
                _, loss_step = sess.run(
                    [self.train_op, self.loss], feed_dict=feed)

                loss_step = np.sum(loss_step)
                mean_loss.append(loss_step)
                batch_count += 1
                if batch_count % 1000 == 0:
                    print('step %d / %d : time: %ds, loss : %f' %
                          (batch_count, len(X_train) // self.batch_size,
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
class Config(object):
    def __init__(self):
        self.batch_size = 1024
        self.hidden_size = 64
        self.vocab_size = 26
        self.embed_size = 512
        self.max_epochs = 40
        self.label_kinds = 2
        self.if_train = False
        self.if_test = False
        self.if_inference = False
        self.is_biLSTM = True
        self.max_seqlen = 20

        self.original_file = '../data/zhwiki-latest-pages-articles.txt'
        self.train_file = '../data/train.txt'
        self.dev_file = '../data/dev.txt'
        self.vocab_file = '../data/vocab.txt'
        self.model_path = 'models/bilstm/'

        self.split_ratio = 0.9
