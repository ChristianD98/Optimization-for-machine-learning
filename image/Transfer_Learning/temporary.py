import numpy as np
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from data import dataset
from transfer_learning import rank_data_according_to_score

dataset.data_dir = 'Bonjour'

# Paths to cached transfer values
file_path_cache_train = os.path.join(dataset.data_dir, 'vgg16_mnist_train.pkl')
file_path_cache_test = os.path.join(dataset.data_dir, 'vgg16_mnist_test.pkl')

# Load transfer values
with open(file_path_cache_train, "rb") as f:
    transfer_values_train = pickle.load(f)
with open(file_path_cache_test, "rb") as f:
    transfer_values_test = pickle.load(f)

# Load labels
y_train_labels = np.argmax(dataset.y_train, axis=1)
y_test_labels = np.argmax(dataset.y_test, axis=1)

# Use a fast SVM
base_clf = LinearSVC(max_iter=10000)
clf = CalibratedClassifierCV(base_clf, cv=3)

print("Fitting LinearSVC...")
clf.fit(transfer_values_train, y_train_labels)

print("Evaluating on test set...")
test_scores = clf.predict_proba(transfer_values_test)
train_scores = clf.predict_proba(transfer_values_train)

print("Test accuracy:", np.mean(np.argmax(test_scores, axis=1) == y_test_labels))

# Rank training samples by difficulty
sorted_indices = rank_data_according_to_score(train_scores, y_train_labels, reverse=True)
np.save("Bonjour/vgg16_transfer_difficulty.npy", sorted_indices)