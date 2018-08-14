## Submission.py for COMP6714-Project2
###################################################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import spacy
import gensim
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    #Process raw inputs into a dataset
    def build_dataset(words,vocabulary_size):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary
    vocabulary_size = 12000 ################################
    processed_data = get_data()
    data, count, dictionary, reverse_dictionary = build_dataset(processed_data,vocabulary_size)

    print('Most common words (+UNK)', count[:5])
    print(len(reverse_dictionary))
    global data_index
    data_index = 0
    def generate_batch(batch_size, num_skips, skip_window):
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels
    #####################
    batch_size = 128
    embedding_size = embedding_dim
    skip_window = 2
    num_skips = 4
    valid_size = 20
    valid_window = 100
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64
    #####################
    graph = tf.Graph()
    with graph.as_default():
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
            softmax_weights = tf.Variable(\
                tf.truncated_normal([vocabulary_size, embedding_size],\
                                    stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            loss = tf.reduce_mean(\
                tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,\
                                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
            with tf.name_scope('Adam_Optimizer'):
                ##learning rate
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    total_steps = num_steps
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(total_steps):
            batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
    #adjective_embeddings.txt
    log = open(embeddings_file_name,'w')
    string = str(len(final_embeddings))+' '+str(embedding_size)+'\n'
    log.write(string)
    for i in range(0,len(final_embeddings)):
        log.write(reverse_dictionary[i]+' '+' '.join(map(str,final_embeddings[i]))+'\n')
    log.close()
    print(len(final_embeddings))

def get_data():
    with open('processed_data.txt','r',encoding='utf8') as f:
        line = f.read()
        data = line.split()
    return data

def process_data(input_data):
    nlp = spacy.load("en")
    #write the raw processed data
    log = open('processed_data.txt','w')
    data=[]
    with zipfile.ZipFile(input_data) as f:
        for filename in f.namelist():
            if not os.path.isdir(filename):
                with f.open(filename) as z:
                    for line in z:
                        test_doc = nlp(str(line.decode('utf-8').rstrip('\n')))
                        for token in test_doc:
                            if token.pos_=='ADJ' or token.pos_=='NOUN':
                                data.append(str(token))
                                log.write(str(token)+'\t')

    log.close()
    return 'processed_data.txt'

def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    similar = model.most_similar(positive = input_adjective,topn=12000)
    result = []
    n=0
    for word in result:
        if count == top_k:
            break
        adj_ = word[0]
        token = nlp(adj_)[0]
        if token.pos_ == 'ADJ':
            result.append(adj_)
            n+=1
    return result








        
