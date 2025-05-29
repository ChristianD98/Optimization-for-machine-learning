from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, ReLU, Lambda
import tensorflow as tf

import numpy as np

class IMDB_Model():
    def build_classifier_model(self, embedding_matrix, input_len=200, hidden_dim=128, lstm_layers=2):
        max_words, embedding_dim = embedding_matrix.shape

        inputs = Input(shape=(input_len,))

        x = Embedding(input_dim=max_words,
                      output_dim=embedding_dim,
                      weights=[embedding_matrix],
                      trainable=False)(inputs)

        for i in range(lstm_layers):
            return_seq = (i < lstm_layers - 1)
            x = LSTM(hidden_dim, return_sequences=return_seq)(x)

        x = Dropout(0.5)(x)
        x = Dense(257)(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)

        return Model(inputs, x)