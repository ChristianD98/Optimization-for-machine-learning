# text_data.py

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from types import SimpleNamespace

# === Global parameters === #
vocab_size = 10000
maxlen = 200
num_classes = 2

# === Load and preprocess the IMDb dataset === #
def load_imdb_dataset(vocab_size=10000, maxlen=200, num_classes=2):
    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=vocab_size)
    
    # Pad sequences to fixed length
    X_train = pad_sequences(train_X, maxlen=maxlen, padding="post", truncating="post")
    X_test = pad_sequences(test_X, maxlen=maxlen, padding="post", truncating="post")
    
    # Convert labels to one-hot vectors
    Y_train = to_categorical(train_y, num_classes=num_classes)
    Y_test = to_categorical(test_y, num_classes=num_classes)

    return SimpleNamespace(
        name="IMDB",
        x_train=X_train,
        y_train=Y_train,
        y_train_labels=np.argmax(Y_train, axis=1),
        x_test=X_test,
        y_test=Y_test,
        y_test_labels=np.argmax(Y_test, axis=1),
        n_classes=num_classes,
    )

# === Get IMDb word index (word → token ID) === #
def get_word_index():
    return imdb.get_word_index()

# === Load GloVe and build the embedding matrix === #
def load_glove_embedding_matrix(glove_path, word_index, vocab_size=10000, embedding_dim=200):
    # Adjust index to account for reserved tokens
    index_word = {v+3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<OOV>"

    # Load GloVe vectors
    embedding_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != embedding_dim + 1:
                continue
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector

    # Build the embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        word = index_word.get(i)
        if word:
            vector = embedding_index.get(word)
            if vector is not None:
                embedding_matrix[i] = vector

    return embedding_matrix
