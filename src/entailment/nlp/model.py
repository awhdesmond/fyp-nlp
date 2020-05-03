import json
import codecs
from os import path
from typing import List, Dict

import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Dense, Input, Dropout, Lambda, Layer, LSTM, Bidirectional, concatenate

import log
logger = log.init_stream_logger(__name__)

##################
# Helper Methods #
##################

def cosine_distance(y1, y2):
    mult = tf.multiply(y1, y2)
    cosine_numerator = tf.reduce_sum(mult, axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), 1e-6))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), 1e-6))
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(text_vector, hypo_vector):
    text_vector_tmp = tf.expand_dims(text_vector, 1)
    hypo_vector_tmp = tf.expand_dims(hypo_vector, 2)
    relevancy_matrix = cosine_distance(text_vector_tmp, hypo_vector_tmp)
    return relevancy_matrix

def max_mean_pooling(repres, cosine_matrix):
    repres.append(tf.reduce_max(cosine_matrix, axis=2, keep_dims=True))
    repres.append(tf.reduce_mean(cosine_matrix, axis=2, keep_dims=True))
    return repres

def matching_layer(inputs):
    forward_relevancy_matrix = cal_relevancy_matrix(inputs[0], inputs[2])
    backward_relevancy_matrix = cal_relevancy_matrix(inputs[1], inputs[3])

    representation = []

    max_mean_pooling(representation, forward_relevancy_matrix)
    max_mean_pooling(representation, backward_relevancy_matrix)

    return representation


class MatchLayer(Layer):
    """
    MatchLayer
    """

    def __init__(self, dim, seq_length, **kwargs):
        super(MatchLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.dim = dim
        self.seq_length = seq_length

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('MatchLayer layer should be called on a list of inputs')

        if all([shape is None for shape in input_shape]):
            return

        super(MatchLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return matching_layer(inputs)

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        output_shape = list(input_shapes[0])

        return [(None, output_shape[1], 1)] * 4

    def get_config(self):
        config = {
            'dim': self.dim,
            'seq_length': self.seq_length
        }
        base_config = super(MatchLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPoolingLayer(Layer):
    """
    MaxPoolingLayer
    """

    def __init__(self, **kwargs):
        super(MaxPoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxPoolingLayer, self).build(input_shape)

    def call(self, inputs):
        return max_mean_pooling([], inputs)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        return [(None, output_shape[1], 1)] * 2

    def compute_mask(self, inputs, mask):
        return [mask, mask]


class TextualEntailmentModel:
    """
    Entailment model for predicting the entailment relationship between
    a hypothesis and a premise

    Attributes:
        data_folder: the base folder for storing all data and model files
        allnli_paths: AllNLI dataset
        rte_paths: RTE dataset
        model_weights: pre-trained model weights
        embeddings_file: GLoVE embeddings file
    """

    labels_map = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2
    }

    def __init__(
        self,
        data_folder: str,
        allnli_paths: List[str],
        rte_paths: List[str],
        model_weights: str,
        embeddings_file: str
    ):
        self.data_folder = data_folder
        self.allnli_paths = allnli_paths
        self.rte_paths = rte_paths
        self.model_weights = model_weights
        self.embeddings_file = embeddings_file

        self.tokenizer = None
        self.max_seq_len = 0

    @property
    def word_embedding_path(self):
        return path.join(self.data_folder, self.embeddings_file)

    @property
    def allnli_data_paths(self):
        return [
            path.join(self.data_folder, allnli_path)
            for allnli_path in self.allnli_paths
        ]

    @property
    def rte_data_paths(self):
        return [
            path.join(self.data_folder, rte_path)
            for rte_path in self.rte_paths
        ]

    def load_word_embeddings(self):
        """
        Load GLoVE word embeddings file

        Returns a dict of {str: [float]}
        """

        logger.info(f"Loading word embeddings from {self.word_embedding_path}")

        with codecs.open(self.word_embedding_path + '.vocab', "r") as f_in:
            index2word = [line.strip() for line in f_in]
        wv = np.load(self.word_embedding_path + '.npy')

        embeddings = {}
        for i, w in enumerate(index2word):
            embeddings[w] = wv[i]
        return embeddings

    def load_data(self):
        """
        Load the data set from different datasets

        Returns a list of strings which are sentences in our corpus
        """
        premises = []
        hypothesis = []

        for data_path in self.allnli_data_paths:
            with open(data_path, "r") as jsonfile:
                logger.info(f"Loading ALLNLI dataset: {data_path}")
                for line in jsonfile:
                    datum = json.loads(line)
                    if datum["gold_label"] not in TextualEntailmentModel.labels_map:
                        continue
                    premises.append(datum["sentence1"])
                    hypothesis.append(datum["sentence2"])

        for data_path in self.rte_data_paths:
            with open(data_path, "r") as jsonfile:
                logger.info(f"Loading RTE dataset: {data_path}")
                jsonObj = json.load(jsonfile)
                for item in jsonObj['pair']:
                    premises.append(item['t'])
                    hypothesis.append(item['h'])

        return premises, hypothesis

    def create_word_embedding_matrix(
        self,
        word_embeddings: Dict,
        word_map: Dict,
        num_words_in_corpus: int,
        embedding_dimension: int,
    ):
        """
        Creates a word embedding matrix mapping word index
        to its embedding vector
        """
        logger.info("Creating word embedding matrix")

        matrix = np.random.random((num_words_in_corpus + 1, embedding_dimension))
        for word, idx in word_map.items():
            embedding_vec_for_word = word_embeddings.get(word)
            if embedding_vec_for_word is not None:
                matrix[idx] = embedding_vec_for_word

        return matrix

    def create_model(self):
        """
        Builds the BiMPM entailment model
        """

        # 1. Load our corpus
        premises, hypotheses = self.load_data()

        # 2. Load the word embeddings
        word_embeddings = self.load_word_embeddings()
        num_words_in_embeddings = len(word_embeddings)
        embedding_dimension = 300

        allSentences = premises + hypotheses

        # 3. Initialise Keras Text Tokenizer
        self.tokenizer = Tokenizer(num_words=num_words_in_embeddings)
        self.tokenizer.fit_on_texts(allSentences)

        # 4. Creates the word embedding matrix
        word_index = self.tokenizer.word_index
        num_words_in_corpus = min(num_words_in_embeddings, len(word_index))

        word_embedding_matrix = self.create_word_embedding_matrix(
            word_embeddings,
            word_index,
            num_words_in_corpus,
            embedding_dimension
        )

        # 5. Find the max seq len of the corpus
        prem_seqs = self.tokenizer.texts_to_sequences(premises)
        hypo_seqs = self.tokenizer.texts_to_sequences(hypotheses)
        longest_prem = 0
        for sentence in prem_seqs:
            longest_prem = max(longest_prem, len(sentence))
        longest_hypo = 0
        for sentence in hypo_seqs:
            longest_hypo = max(longest_hypo, len(sentence))
        self.max_seq_len = max(longest_prem, longest_hypo)

        logger.info(f"Max Seq Length: {self.max_seq_len}")

        premise_input = Input(shape=(self.max_seq_len,), dtype='int32', name='premise')
        hypo_input = Input(shape=(self.max_seq_len,), dtype='int32', name='hypo')

        def word_context(_input, name):
            embedding = Embedding(
                num_words_in_corpus + 1,
                embedding_dimension,
                weights=[word_embedding_matrix],
                input_length=self.max_seq_len,
                trainable=False,
                name=name+"_embedding"
            )(_input)

            word = Dropout(0.1)(embedding)
            context = Bidirectional(
                LSTM(100, return_sequences=True),
                merge_mode=None,
                name=name + '_context'
            )(word)
            return (word, context)

        (premiseEmbedding, premiseContext) = word_context(premise_input, 'text')
        (hypoEmbedding, hypoContext) = word_context(hypo_input, 'hypothesis')

        left_context = []
        left_context.extend(hypoContext)
        left_context.extend(premiseContext)

        left_match = MatchLayer(embedding_dimension, self.max_seq_len)(left_context)

        right_context = []
        right_context.extend(premiseContext)
        right_context.extend(hypoContext)

        right_match = MatchLayer(embedding_dimension, self.max_seq_len)(right_context)

        cosine_left = Lambda(lambda x_input: cal_relevancy_matrix(x_input[0], x_input[1]))(
            [premiseEmbedding, hypoEmbedding]
        )
        consine_right = Lambda(lambda cosine: tf.transpose(cosine, perm=[0, 2, 1]))(cosine_left)

        left_repr = MaxPoolingLayer()(cosine_left)
        right_repr = MaxPoolingLayer()(consine_right)

        left_repr.extend(left_match)
        right_repr.extend(right_match)

        left = concatenate(left_repr, axis=2)
        left = Dropout(0.1)(left)

        right = concatenate(right_repr, axis=2)
        right = Dropout(0.1)(right)

        agg_left = Bidirectional(LSTM(100), name='aggregation_premise_context')(left)
        agg_right = Bidirectional(LSTM(100), name='aggregation_hypo_context')(right)
        aggregation = concatenate([agg_left, agg_right], axis=-1)

        pred = Dense(200, activation='tanh', name='tanh_prediction')(aggregation)
        pred = Dense(3, activation='softmax', name='softmax_prediction')(pred)

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model = Model(inputs=[premise_input, hypo_input], outputs=pred)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        logger.info("Loading model weights")
        model.load_weights(path.join(self.data_folder, self.model_weights))
        logger.info("Model weights loaded")

        global keras_graph
        keras_graph = tf.get_default_graph()

        self.model = model

    def predict(self, premise: str, hypothesis: str):
        """
        Predicts the entailment relationship between the given premise and hypothesis

        Args:
            premise:
            hypothesis

        Returns the list of scores for each entailment label
        """

        premise_seq = self.tokenizer.texts_to_sequences([premise])
        hypo_seq = self.tokenizer.texts_to_sequences([hypothesis])

        prem_data = pad_sequences(premise_seq, maxlen=self.max_seq_len)
        hypo_data = pad_sequences(hypo_seq, maxlen=self.max_seq_len)

        with keras_graph.as_default():
            return self.model.predict([prem_data, hypo_data])[0]
