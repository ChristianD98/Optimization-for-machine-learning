from keras.datasets import mnist
import numpy as np
from types import SimpleNamespace
from keras.utils import to_categorical

(train_X, train_y), (test_X, test_y) = mnist.load_data()


X_train, Y_train = train_X, train_y
X_test, Y_test = test_X, test_y

X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)


dataset = SimpleNamespace(
    name="MNIST",
    x_train=X_train,
    y_train=Y_train,
    y_train_labels=np.argmax(Y_train, axis=1),
    x_test=X_test,
    y_test=Y_test,
    y_test_labels=np.argmax(Y_test, axis=1),
    n_classes=10,
)


print('X_train: ' + str(dataset.x_train.shape))
print('Y_train: ' + str(dataset.y_train.shape))
print('X_test:  '  + str(dataset.x_test.shape))
print('Y_test:  '  + str(dataset.y_test.shape))
