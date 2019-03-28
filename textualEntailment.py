import collections
import json
import os
from os.path import join
import time

import numpy as np

from tqdm import tqdm

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Dropout, Reshape, BatchNormalization, TimeDistributed, Lambda, Layer, LSTM, Bidirectional, add, concatenate
from keras.optimizers import RMSprop, Adam, SGD, Adagrad
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, BaseLogger, ReduceLROnPlateau

# CONSTANTS
lr = 0.001
lr_decay = 1e-4
epochs = 5
batch_size = 16

eps = 1e-6

DATA_FOLDER = "./AllNLI"

# Not used
ALLNLI_DEV_PATH = "./AllNLI/dev.jsonl"
ALLNLI_TRAIN_PATH = "./AllNLI/train.jsonl"
ALLNLI_TEST_PATH = "./AllNLI/test.jsonl"

SNLI_DEV_PATH = "./snli_1.0/snli_1.0_dev.jsonl"
SNLI_TRAIN_PATH = "./snli_1.0/snli_1.0_train.jsonl"
SNLI_TEST_PATH ="./snli_1.0/snli_1.0_test.jsonl"

# Using
MULTINLI_MATCH_PATH = join(DATA_FOLDER, "./multinli_1.0/multinli_1.0_dev_matched.jsonl") 
MULTINLI_MISMATCH_PATH = join(DATA_FOLDER, "./multinli_1.0/multinli_1.0_dev_mismatched.jsonl") 
MULTINLI_TRAIN_PATH = join(DATA_FOLDER, "./multinli_1.0/multinli_1.0_train.jsonl")  

RTE_DEV_PATH = join(DATA_FOLDER, "./rte_1.0/rte.json")
RTE_TEST_PATH = join(DATA_FOLDER, "./rte_1.0/rte_test.json") 

FASTTEXT_FILE = join(DATA_FOLDER, "fasttext-crawl-300d-2M-subword.vec")
MODEL_WEIGHTS = join(DATA_FOLDER, "fasttext-full-fasttext-full-adam-weights.h5")

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def cosine_distance(y1, y2):
    mult =  tf.multiply(y1, y2)
    cosine_numerator = tf.reduce_sum( mult, axis = -1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1 ), eps) ) 
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1 ), eps) ) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(text_vector, hypo_vector):
    text_vector_tmp = tf.expand_dims(text_vector, 1) # [batch_size, 1, question_len, dim]
    hypo_vector_tmp = tf.expand_dims(hypo_vector, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(text_vector_tmp, hypo_vector_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, text_mask, hypo_mask):
    relevancy_matrix = tf.multiply(relevancy_matrix, K.expand_dims(text_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, K.expand_dims(hypo_mask, 2))
    return relevancy_matrix

def max_mean_pooling(repres, cosine_matrix):
    
    repres.append(tf.reduce_max(cosine_matrix, axis = 2, keep_dims = True))
    repres.append(tf.reduce_mean(cosine_matrix, axis = 2, keep_dims = True))

    return repres

def matching_layer(inputs):
    forward_relevancy_matrix = cal_relevancy_matrix(inputs[0], inputs[2])
    backward_relevancy_matrix = cal_relevancy_matrix(inputs[1], inputs[3])

    representation = []

    max_mean_pooling(representation, forward_relevancy_matrix)
    max_mean_pooling(representation, backward_relevancy_matrix)
    
    return representation

##
## MatchLayer
##
class MatchLayer(Layer):

    def __init__(self, dim, seq_length, **kwargs):
        super(MatchLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.dim = dim
        self.seq_length = seq_length
        
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('`MatchLayer` layer should be called '
                             'on a list of inputs')
        
        if all([shape is None for shape in input_shape]):
            return
        
        super(MatchLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatchLayer` layer should be called ')
        
        return matching_layer(inputs)
    
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `MatchLayer` layer should be called '
                             'on a list of inputs.')
        
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
                             
        return [ (None, output_shape[1] , 1) ] * 4 
    
    def get_config(self):
        config = {
            'dim': self.dim,
            'seq_length': self.seq_length
        }
        base_config = super(MatchLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

##
## MaxPoolingLayer
##
class MaxPoolingLayer(Layer):

    def __init__(self, **kwargs):
        super(MaxPoolingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MaxPoolingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return max_mean_pooling([], inputs)
    
    def compute_output_shape(self, input_shape):            
        output_shape = list(input_shape)
        return [ (None, output_shape[1] , 1) ] * 2
    
    def compute_mask(self, inputs, mask):
        return [mask, mask]

##
## TextualEntailmentModel
##
class TextualEntailmentModel(object):

    labels =  ["entailment", "contradiction", "neutral"]
    labelsMap = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.maxSeqLen = 0

    def loadDataset(self):
        RTE_PATHS = [RTE_DEV_PATH, RTE_TEST_PATH]
        MNLI_PATHS = [MULTINLI_MATCH_PATH, MULTINLI_MISMATCH_PATH, MULTINLI_TRAIN_PATH]
        
        premises = []
        hypothesis = []
        labels = []

        for path in RTE_PATHS:
            with open(path, 'r') as jsonfile:
                jsonObj = json.load(jsonfile)
                for item in tqdm(jsonObj['pair']):
                    datum = {
                        'sentence1': item['t'],
                        'sentence2': item['h'],
                    }
                    if item['-entailment'] == "YES":
                        datum['gold_label'] = "entailment"
                    elif item['-entailment'] == "NO":
                        datum['gold_label'] = "contradiction"
                    else:
                        datum['gold_label'] = "neutral"
                    
                    label = datum["gold_label"]
                    premises.append(datum["sentence1"])
                    hypothesis.append(datum["sentence2"])
                    labels.append(self.labelsMap[label])
        
        for path in MNLI_PATHS:
            with open(path, 'r') as jsonfile:
                for line in tqdm(jsonfile):
                    datum = json.loads(line)

                    label = datum["gold_label"]
                    if label in self.labelsMap:
                        premises.append(datum["sentence1"])
                        hypothesis.append(datum["sentence2"])
                        labels.append(self.labelsMap[label])

                        
        return premises, hypothesis, labels

    def loadWordEmbeddings(self):    
        fasttextFile = FASTTEXT_FILE
        wordEmbeddings = {}

        with open(fasttextFile, 'r') as file:
            for line in tqdm(file):
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float32")
                wordEmbeddings[word] = embedding

        return wordEmbeddings

    def getEmbeddingDim(self, wordEmbeddings):
        return len(wordEmbeddings['a'])

        
    def createWordEmbeddingMatrix(self, wordEmbeddings, wordMap, numWordsInCorpus, embeddingDim):
        wordEmbeddingMatrix = np.random.random((numWordsInCorpus + 1, embeddingDim))

        validEmbeddings = 0
        for word, idx in wordMap.items():
            embeddingVectorForWord = wordEmbeddings.get(word)
            if embeddingVectorForWord is not None:
                wordEmbeddingMatrix[idx] = embeddingVectorForWord
                validEmbeddings += 1
        
        return wordEmbeddingMatrix

    def createModel(self):
        (premiseData, hypothesisData, labelsData) = self.loadDataset()
        wordEmbeddings = self.loadWordEmbeddings()
        embeddingDimension = self.getEmbeddingDim(wordEmbeddings)

        numWords = len(wordEmbeddings)
        allSentences = premiseData + hypothesisData

        # Create word map from corpus (i.e training data)
        self.tokenizer = Tokenizer(num_words=numWords)
        self.tokenizer.fit_on_texts(allSentences)
        premisesWordSequences = self.tokenizer.texts_to_sequences(premiseData)
        hypothesisWordSequences = self.tokenizer.texts_to_sequences(hypothesisData)

        wordMap = self.tokenizer.word_index # mapping of a word to its index

        numWordsInCorpus = min(numWords, len(wordMap))

        wordEmbeddingMatrix = self.createWordEmbeddingMatrix(wordEmbeddings, wordMap, numWordsInCorpus, embeddingDimension)

        longestPremiseWordCount = 0
        for sentence in premisesWordSequences:
            longestPremiseWordCount = max(longestPremiseWordCount, len(sentence))

        longestHypoWordCount = 0
        for sentence in hypothesisWordSequences:
            longestHypoWordCount = max(longestHypoWordCount, len(sentence))
            
        self.maxSeqLen = max(longestPremiseWordCount, longestHypoWordCount)

        premiseInput = Input(shape=(self.maxSeqLen,), dtype='int32', name='premise')
        hypoInput = Input(shape=(self.maxSeqLen,), dtype='int32', name='hypo')

        def wordContext(_input, name):
            embedding = Embedding(numWordsInCorpus + 1,
                                  embeddingDimension,
                                  weights=[wordEmbeddingMatrix], 
                                  input_length=self.maxSeqLen, 
                                  trainable=False, 
                                  name=name+"_embedding")(_input)


            word = Dropout(0.1)(embedding)
            context = Bidirectional(LSTM(100, return_sequences=True),
                                    merge_mode=None,
                                    name = name + '_context')(word)
            return (word, context)
        
        (premiseEmbedding, premiseContext) = wordContext(premiseInput, 'text')
        (hypoEmbedding, hypoContext) = wordContext(hypoInput, 'hypothesis')

        leftContext = []
        leftContext.extend(hypoContext)
        leftContext.extend(premiseContext)
        
        leftMatch = MatchLayer(embeddingDimension, self.maxSeqLen)(leftContext)
        
        rightContext = []
        rightContext.extend(premiseContext)
        rightContext.extend(hypoContext)
        
        rightMatch = MatchLayer(embeddingDimension, self.maxSeqLen)(rightContext)
        
        cosineLeft = Lambda(lambda x_input: cal_relevancy_matrix(x_input[0], x_input[1]))([premiseEmbedding, hypoEmbedding])
        cosineRight = Lambda(lambda cosine: tf.transpose(cosine, perm=[0,2,1]))(cosineLeft)
        
        leftRepresentation = MaxPoolingLayer()(cosineLeft)
        rightRepresentation = MaxPoolingLayer()(cosineRight)
        
        leftRepresentation.extend(leftMatch)
        rightRepresentation.extend(rightMatch)
        
        left = concatenate(leftRepresentation, axis = 2)
        left = Dropout(0.1)(left)
        
        right = concatenate(rightRepresentation, axis = 2)
        right = Dropout(0.1)(right)
        
        aggregationLeft = Bidirectional(LSTM(100), name='aggregation_premise_context')(left)

        aggregationRight = Bidirectional(LSTM(100), name='aggregation_hypo_context')(right)
        
        aggregation = concatenate([aggregationLeft, aggregationRight], axis = -1)
                                
        pred = Dense(200, activation = 'tanh', name = 'tanh_prediction')(aggregation)
        pred = Dense(3, activation = 'softmax', name = 'softmax_prediction')(pred)
        
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        model = Model(inputs=[premiseInput, hypoInput], outputs=pred)
        model.compile(loss = 'binary_crossentropy', 
                      optimizer = optimizer,
                      metrics = ['accuracy'])
        
        print('Model created')
        
        model.load_weights(MODEL_WEIGHTS)
        self.model = model

    def predict(self, premise, hypothesis):
        s1_word_sequence = self.tokenizer.texts_to_sequences([premise])
        s2_word_sequence = self.tokenizer.texts_to_sequences([hypothesis])

        s1_data = pad_sequences(s1_word_sequence, maxlen=self.maxSeqLen)
        s2_data = pad_sequences(s2_word_sequence, maxlen=self.maxSeqLen)

        return self.model.predict([s1_data, s2_data])[0]