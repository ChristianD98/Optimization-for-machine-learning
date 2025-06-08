from keras.models import Model
from keras.layers import Input, Dense, Dropout, ReLU
import tensorflow as tf

class IMDB_Model:
    def build_mlp_model(self, embedding_dim=384):
        
        inputs = Input(shape=(embedding_dim,))
        x = Dense(256)(inputs)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(128)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(64)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(32)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(2, activation='softmax')(x)

        return Model(inputs, x)