#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model for ESC-50 classification using pre-computed CLAP embeddings
"""

from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import Dropout, Dense, BatchNormalization, Activation, Input
import ModelLib

class ESC50_Embedding_Model(ModelLib.ModelLib):
    def build_classifier_model(self, dataset, n_classes=50,
                              activation='relu', dropout_rate=0.5,
                              reg_factor=1e-4, batch_norm=True):
        """
        Build a classifier model for pre-computed CLAP embeddings on ESC-50
        
        Args:
            dataset: The ESC50 dataset object
            n_classes: Number of classes (defaults to 50 for ESC-50)
            activation: Activation function to use
            dropout_rate: Dropout rate
            reg_factor: L2 regularization factor
            batch_norm: Whether to use batch normalization
            
        Returns:
            A compiled Keras model
        """
        # Set the number of classes from the dataset if provided
        if hasattr(dataset, 'n_classes'):
            n_classes = dataset.n_classes
            
        # L2 regularization
        l2_reg = regularizers.l2(reg_factor)
        
        # Using the shape from our dataset embeddings
        input_shape = dataset.x_train.shape[1:]
        
        # Create model for embeddings (which are feature vectors)
        x = input_layer = Input(shape=input_shape)
        
        # First dense layer
        x = Dense(256, kernel_regularizer=l2_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        
        # Second dense layer
        x = Dense(128, kernel_regularizer=l2_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        x = Dense(n_classes, kernel_regularizer=l2_reg)(x)
        x = Activation('softmax')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=x)
        
        return model
