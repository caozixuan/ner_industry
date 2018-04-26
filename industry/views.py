import gensim
import numpy as np
import os
import tensorflow as tf
import jieba
import random
import re
from industry.models import Word, WordTest

# keyword used to locate industry entity
rules = [r'(.*)上游(.*)', r'下游(.*)', r'(.*)应用于(.*)']
keywords = ['上游', '下游', '应用']

# you should change the path to your own  model path
word2vec_model_path = 'D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx'

# you should change the path to your own data path
data_path = 'F:\\tmp_data\\temp_data\\data'

# parameters used while training
sentence_length = 15
word_dim = 256
class_size = 2
rnn_size = 256
num_layers = 2
epoch = 80
batch_size = 20
learning_rate = 0.003


# build your own word2vec
def train_data_build():
    file = r'F:\train_data.txt'
    names = file_name('F:\\data')
    for name in names:
        f = open(name, errors='ignore')
        st = f.read()
        with open(file, 'a+') as f:
            seg_list = jieba.cut(st, cut_all=False)
            f.write(" ".join(seg_list))
            f.write('\n')
        f.close()


def train_data():
    from gensim.models import word2vec
    sentences = word2vec.Text8Corpus('F:\\train_data.txt')
    model = word2vec.Word2Vec(sentences, size=50)
    model.save('word2vec_model')


# get the names of document from the path
def file_name(file_dir):
    names_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                names_list.append(os.path.join(root, file))
    return names_list


# collect the needed data from documents to database. True means collect training data,False means collect test data
def collect_data(flag=True):
    if flag:
        total_file = 200
    else:
        total_file = 50
    names = file_name(data_path)
    names_slice = random.sample(names, total_file)
    sentence_id = 0
    for name in names_slice:
        word_counter = 0
        f = open(name, errors='ignore')
        st = f.read()
        for j in range(0, len(rules)):
            compile_name = re.compile(rules[j], re.M)
            res_name = compile_name.finditer(st)
            for m in res_name:
                seg_list = jieba.lcut(m.group(), cut_all=False)
                target_list = []
                for w in seg_list:
                    if len(w) > 1:
                        target_list.append(w)
                print(target_list)
                try:
                    target = target_list.index(keywords[j])
                except ValueError:
                    continue
                words_in_sentence = []
                index = target - 7
                for i in range(15):
                    if index >= 0:
                        try:
                            words_in_sentence.append(target_list[index])
                        except IndexError:
                            words_in_sentence.append("#")
                    else:
                        words_in_sentence.append("#")
                    index += 1
                for word in words_in_sentence:
                    if flag:
                        Word.objects.create(name=word, sentence_id=sentence_id)
                    else:
                        WordTest.objects.create(name=word, sentence_id=sentence_id)
                sentence_id += 1
            word_counter += 1


# delete the word that don't have a word2vec
def delete_word():
    words_train = Word.objects.all()
    words_test = WordTest.objects.all()
    words = words_train + words_test
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    for word in words:
        if word.name == '#':
            continue
        try:
            model[word.name]
        except  KeyError:
            print(word.name)
            Word.objects.filter(name=word.name).update(name='#')


# delete sentences that do not have any industry entities
def delete_data():
    for counter in range(0, 250):
        words = Word.objects.filter(sentence_id=counter)
        flag = True
        for word in words:
            if word.div_type != 0:
                flag = False
        if flag:
            Word.objects.filter(sentence_id=counter).delete()


class Model:
    def __init__(self):
        self.input_data = tf.placeholder(tf.float32, [None, sentence_length, word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, sentence_length, class_size])
        fw_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        output, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                      tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                                      dtype=tf.float32, sequence_length=self.length)
        weight, bias = self.weight_and_bias(2 * rnn_size, class_size)
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, sentence_length, class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


# evaluate the prediction of the model
def f1(prediction, target, length):
    tp = np.array([0] * (class_size + 1))
    fp = np.array([0] * (class_size + 1))
    fn = np.array([0] * (class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    for i in range(class_size):
        tp[class_size] += tp[i]
        fp[class_size] += fp[i]
        fn[class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[class_size - 1]


def get_train_test_data():
    zero = []
    model = gensim.models.Word2Vec.load('D:\\word2vec\\word2vec_from_weixin\\word2vec\\word2vec_wx')
    for i in range(0, 256):
        zero.append(0.1)
    words = Word.objects.all()
    i = 0
    counter = 0
    inputs = []
    outputs = []
    word_sentence = []
    while i < len(words):
        input = np.empty((15, 256))
        output = np.empty((15, class_size))
        word_test = []
        tmp_words = words[i:i + 15]
        print(tmp_words)
        j = 0
        for word in tmp_words:
            try:
                input[j] = model[word.name]
                word_test.append(word.name)
            except KeyError:
                input[j] = np.array(zero)
                word_test.append(word.name)
            if word.div_type == 0:
                tmp = [1, 0]
            elif word.div_type == 1:
                tmp = [0, 1]
            else:
                tmp = [0, 1]
            output[j] = np.array(tmp)
            counter += 1
            j += 1
        inputs.append(input)
        outputs.append(output)
        word_sentence.append(word_test)
        i += 15
    split_point = int((len(words) / 15) * 0.8)
    train_input = inputs[0:split_point]
    train_output = outputs[0:split_point]
    test_input = inputs[split_point:(len(words) / 15) - 1]
    test_output = outputs[split_point:(len(words) / 15) - 1]
    word_sentence = word_sentence[split_point:(len(words) / 15) - 1]
    return np.array(train_input), np.array(train_output), np.array(test_input), np.array(test_output), word_sentence


def train():
    train_inp, train_out, test_a_inp, test_a_out, word_sentence = get_train_test_data()
    for sentence in word_sentence:
        print(sentence)
    model = Model()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for e in range(epoch):
            for ptr in range(0, len(train_inp), batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + batch_size],
                                          model.output_data: train_out[ptr:ptr + batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, "./model/model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out})
            print("epoch %d:" % e)
            print('test_a score:')
            f1(pred, test_a_out, length)
            for z in pred:
                print(z)
