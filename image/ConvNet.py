from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Input



class MNIST_Model():
    def build_classifier_model(self, dataset, n_classes=10, activation='relu', dropout_rate=0.2,reg_factor=50e-4):
        
        n_classes = dataset.n_classes
        
        l2_reg = regularizers.l2(reg_factor)
        
        # input image dimensions
        h, w, d = 28, 28, 1
        input_shape = (h, w, d)

        # input image dimensions
        x = input_1 = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = Activation(activation=activation)(x)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_rate)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = Activation(activation=activation)(x)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_rate)(x)

        x = Flatten()(x)
        x = Dense(units=512, kernel_regularizer=l2_reg)(x)
        x = Activation(activation=activation)(x)

        x = Dropout(rate=dropout_rate)(x)
        x = Dense(units=n_classes, kernel_regularizer=l2_reg)(x)
        x = Activation(activation='softmax')(x)

        model = Model(inputs=[input_1], outputs=[x])
        return model