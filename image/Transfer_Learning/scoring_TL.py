from transfer_learning import get_transfer_values_classic_networks, get_svm_scores, rank_data_according_to_score
from data import dataset
import numpy as np

dataset.data_dir = 'Bonjour'

transfer_values_train, transfer_values_test = get_transfer_values_classic_networks(dataset, network_name="vgg16")

y_train_labels = np.argmax(dataset.y_train, axis=1)
y_test_labels = np.argmax(dataset.y_test, axis=1)

train_scores, test_scores = get_svm_scores(
    transfer_values_train, y_train_labels,
    transfer_values_test, y_test_labels,
    dataset, network_name="vgg16"
)

sorted_indices = rank_data_according_to_score(train_scores, y_train_labels, reverse=True)

np.save("Bonjour/vgg16_transfer_difficulty.npy", sorted_indices)

