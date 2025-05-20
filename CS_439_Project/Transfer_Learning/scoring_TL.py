from transfer_learning import get_transfer_values_classic_networks, get_svm_scores, rank_data_according_to_score
from data import dataset

dataset.data_dir = 'Bonjour'

transfer_values_train, transfer_values_test = get_transfer_values_classic_networks(dataset, network_name="vgg16")

train_scores, test_scores = get_svm_scores(
    transfer_values_train, dataset.y_train,
    transfer_values_test, dataset.y_test,
    dataset, network_name="vgg16"  # or "inception"
)

sorted_indices = rank_data_according_to_score(train_scores, dataset.y_train, reverse=True)

