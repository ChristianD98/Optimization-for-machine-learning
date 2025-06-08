
import numpy as np
from keras.datasets import imdb
from keras.utils import to_categorical
from types import SimpleNamespace

# === Global parameters === #
vocab_size = 10000
num_classes = 2

def load_imdb_labels(vocab_size=10000, num_classes=2):
    (_, train_y), (_, test_y) = imdb.load_data(num_words=vocab_size)

    Y_train = to_categorical(train_y, num_classes=num_classes)
    Y_test = to_categorical(test_y, num_classes=num_classes)

    return SimpleNamespace(
        y_train=Y_train,
        y_test=Y_test,
        y_train_labels=np.argmax(Y_train, axis=1),
        y_test_labels=np.argmax(Y_test, axis=1),
        n_classes=num_classes
    )
