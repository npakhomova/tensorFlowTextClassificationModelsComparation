from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from sklearn import metrics
import datetime

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
import csv

### Training data
def loadData(filename):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        for row in data_file:
            target.append(row.pop(0))
            data.append(row.pop(0))

    target = np.array(target, dtype=np.int32)
    data = np.array(data, dtype=np.str)
    return data, target


### Models


def bag_of_words_model(x, y):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  features = tf.reduce_max(word_vectors, reduction_indices=1)
  prediction, loss = learn.models.logistic_regression(features, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

def rnn_model(x, y):
  """Recurrent neural network model to predict from sequence of words
  to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unpack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for logistic
  # regression over output classes.
  target = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  prediction, loss = learn.models.logistic_regression(encoding, target)

  # Create a training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

def cnn_model(x, y):
  """2 layer Convolutional network to predict from sequence of words
  to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  y = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  word_vectors = tf.expand_dims(word_vectors, 3)
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.contrib.layers.convolution2d(word_vectors, N_FILTERS,
                                            FILTER_SHAPE1, padding='VALID')
    # Add a RELU for non linearity.
    conv1 = tf.nn.relu(conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
        strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.contrib.layers.convolution2d(pool1, N_FILTERS,
                                            FILTER_SHAPE2, padding='VALID')
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  # Apply regular WX + B and classification.
  prediction, loss = learn.models.logistic_regression(pool2, y)

  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


def char_rnn_model(x, y):
  """Character level recurrent neural network model to predict classes."""
  y = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  byte_list = learn.ops.one_hot_matrix(x, 256)
  byte_list = tf.unpack(byte_list, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.rnn(cell, byte_list, dtype=tf.float32)

  prediction, loss = learn.models.logistic_regression(encoding, y)

  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

def char_cnn_model(x, y):
  """Character level convolutional neural network model to predict classes."""
  y = tf.one_hot(y, NUMBER_OF_CATEGORIES, 1, 0)
  byte_list = tf.reshape(learn.ops.one_hot_matrix(x, 256),
                         [-1, MAX_DOCUMENT_LENGTH, 256, 1])
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.contrib.layers.convolution2d(byte_list, N_FILTERS,
                             FILTER_SHAPE1, padding='VALID')
    # Add a RELU for non linearity.
    conv1 = tf.nn.relu(conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
                           strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.contrib.layers.convolution2d(pool1, N_FILTERS,
                                            FILTER_SHAPE2,
                                            padding='VALID')
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  # Apply regular WX + B and classification.
  prediction, loss = learn.models.logistic_regression(pool2, y)

  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


#----testingFunctions

def trainModelAndPrintAccuracy(currectModelDescr):
    global n_words
    print ('Reading Data')
    # Downloads, unpacks and reads DBpedia dataset.
    # dbpedia = learn.datasets.load_dataset('dbpedia')
    X_train, y_train = loadData(currectModelDescr['trainingPath'])
    X_test, y_test = loadData(currectModelDescr['testingPath'])
    ### Process vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    X_train = np.array(list(vocab_processor.fit_transform(X_train)))
    X_test = np.array(list(vocab_processor.transform(X_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)
    classifier = learn.Estimator(model_fn=currectModelDescr['model'],
                                 model_dir=currectModelDescr['modelPath'])
    start = datetime.datetime.now()

    print (str(start)+' Start training')
    classifier.fit(X_train, y_train, steps=TRAINING_STEPS)

    print('Training finished for model: %s in time %s'%(currectModelDescr['name'], str(datetime.datetime.now() - start)))
    print ('Calculate accuracy:')
    y_predicted = [p['class'] for p in classifier.predict(X_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


def classifyGivenText(currectModelDescr, text):
    global n_words
    # Downloads, unpacks and reads DBpedia dataset.
    # dbpedia = learn.datasets.load_dataset('dbpedia')
    X_train, y_train = loadData(currectModelDescr['trainingPath'])
    ### Process vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    X_train = np.array(list(vocab_processor.fit_transform(X_train)))
    n_words = len(vocab_processor.vocabulary_)

    classifier = learn.Estimator(model_fn=currectModelDescr['model'],
                                 model_dir=currectModelDescr['modelPath'])

    ## todo ugly way to restore model!!! no need to pass training set. need to pass empty
    classifier.fit(X_train, y_train, steps=0)


    # np.array(data, dtype=np.str)
    textForClassification = np.array(list(vocab_processor.transform(np.array([text], dtype=np.str))))
    y_predicted = [p['class'] for p in classifier.predict(textForClassification, as_iterable=True)]

    print (y_predicted)

#---PLEAS UPDATE-PATH!!!

MODELS ={
    'bagOfWords' : { 'trainingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'testingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                     'modelPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/modelBagOfWords',
                     'model': bag_of_words_model,
                     'name': 'Bag of Words based on w2vec'},
    'rnn':         { 'trainingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'testingPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                     'modelPath' : '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/rnn_model',
                     'model': rnn_model,
                     'name': 'Recurrent neural network'},
    'cnn':         { 'trainingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'testingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                     'modelPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/cnn_model',
                     'model': cnn_model,
                     'name': '2 layer Convolutional network'},
    'char_rnn':     {'trainingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'testingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                     'modelPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/char_rnn',
                     'model': char_rnn_model,
                     'name': 'Character level recurrent neural network'},
    'char_cnn':     {'trainingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTraining.csv',
                     'testingPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/data/SMALL_MPTTesting.csv',
                      'modelPath': '/Users/npakhomova/PycharmProjects/babyStepsInPython/textClassification/models/char_cnn_model',
                      'model': char_rnn_model,
                      'name': 'Character level convolutional neural network'},
}

#----------MAIN---------

TRAINING_STEPS = 200
NUMBER_OF_CATEGORIES = 29
EMBEDDING_SIZE = 50
MAX_DOCUMENT_LENGTH = 100

# --- parameters for CNN--


N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

#--- char CNN
HIDDEN_SIZE = 20

#-------
trainModelAndPrintAccuracy(MODELS['bagOfWords'])
trainModelAndPrintAccuracy(MODELS['rnn'])

trainModelAndPrintAccuracy(MODELS['cnn'])


# trainModelAndPrintAccuracy(MODELS['char_rnn'])
# trainModelAndPrintAccuracy(MODELS['char_cnn'])

# classifyGivenText(MODELS['bagOfWords'], 'alpine stars hughes hat highlight your laid back look with this stylin  hat from alpine stars')

