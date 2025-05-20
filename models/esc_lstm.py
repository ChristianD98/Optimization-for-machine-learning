#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esc50_lstm_model.py
-------------------
Keras Bi-directional LSTM classifier for the ESC-50 dataset, mirroring the
coding style and API of *cifar100_model.py*.

* Provides a `Esc50_LSTM_Model` class that derives from `ModelLib.ModelLib`.
* Defines a single method `build_classifier_model()` with a signature parallel
  to the CIFAR implementation, allowing easy hyper-parameter tweaking from the
  training script or a notebook.

Note: This file focuses solely on network topology creation. Optimiser,
training loop, callbacks, etc. are expected to be handled by the surrounding
pipeline just like in the original CIFAR codebase.
"""

from keras import regularizers, backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)

import ModelLib  # assumed to be available in the project path


class Esc50_LSTM_Model(ModelLib.ModelLib):
    """Bi-directional LSTM acoustics classifier with optional batch-norm & L2.

    Parameters closely follow *cifar100_model.py* so that existing training
    scripts can swap datasets with minimal code changes.
    """

    def build_classifier_model(
        self,
        dataset,
        n_classes: int | None = None,
        activation: str = "elu",
        dropout_rate: float = 0.3,
        reg_factor: float = 1e-4,
        bias_reg_factor: float | None = None,
        lstm_units: int = 128,
        lstm_layers: int = 2,
        batch_norm: bool = False,
    ) -> Model:
        """Return a compiled *keras.Model* ready for training.

        Parameters
        ----------
        dataset : object
            Must expose ``height`` (mel bins), ``width`` (time frames) and
            ``n_classes`` (≈50) attributes.
        n_classes : int, optional
            Overrides ``dataset.n_classes`` if given.
        activation : str
            Non-linear activation for hidden dense layer.
        dropout_rate : float
            Dropout applied after dense layer and inside LSTM cells.
        reg_factor, bias_reg_factor : float
            L2 regularisation factors (kernel / bias). Bias regularisation is
            disabled by default like in the CIFAR code.
        lstm_units : int
            Number of units in each (bi-)LSTM layer.
        lstm_layers : int
            Stack count of bi-directional LSTM layers.
        batch_norm : bool
            Whether to insert *BatchNormalization* after LSTM outputs and dense
            layers (kept optional to stay faithful to the original template).
        """

        # dataset-derived dimensions
        n_classes = n_classes or getattr(dataset, "n_classes", 50)
        timesteps, n_features = dataset.width, dataset.height

        # regularisers
        l2_reg = regularizers.l2(reg_factor) if reg_factor else None
        l2_bias_reg = (
            regularizers.l2(bias_reg_factor) if bias_reg_factor else None
        )

        # --- model graph --- #
        x = input_1 = Input(shape=(timesteps, n_features))

        for layer_idx in range(lstm_layers):
            return_seq = layer_idx < lstm_layers - 1
            x = Bidirectional(
                LSTM(
                    lstm_units,
                    return_sequences=return_seq,
                    dropout=dropout_rate,
                    kernel_regularizer=l2_reg,
                    bias_regularizer=l2_bias_reg,
                ),
                name=f"bi_lstm_{layer_idx + 1}",
            )(x)
            if batch_norm:
                x = BatchNormalization()(x)

        # fully-connected head
        x = Dense(
            units=128,
            kernel_regularizer=l2_reg,
            bias_regularizer=l2_bias_reg,
        )(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Dropout(rate=dropout_rate)(x)

        x = Dense(
            units=n_classes,
            kernel_regularizer=l2_reg,
            bias_regularizer=l2_bias_reg,
        )(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation("softmax")(x)

        model = Model(inputs=[input_1], outputs=[x], name="esc50_bilstm")
        return model
