#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN+LSTM model for ESC-50 audio classification using mel spectrograms
"""

from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Permute, TimeDistributed, Flatten
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
import ModelLib

def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

class ESC50_Model(ModelLib.ModelLib):
    
    def build_classifier_model(self, dataset, n_classes=50):
        inputs = Input(shape=(dataset.x_train.shape[1:]))

        x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
        x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
        x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
        x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

        x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
        x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
        x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
        x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

        x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
        x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
        x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
        x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

        x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
        x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
        x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
        x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

        x = Add()([x_1, x_2, x_3, x_4])

        x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
        x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

        x = GlobalAveragePooling2D()(x)
        x = Dense(n_classes)(x)
        x = Activation("softmax")(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model