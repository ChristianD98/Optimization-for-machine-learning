import os
from models.inception import transfer_values_cache

from sklearn import svm
import numpy as np
import pickle



def transfer_values_svm_scores(train_x, train_y, test_x, test_y):
    clf = svm.SVC(probability=True)
    print("fitting svm")
    clf.fit(train_x, train_y)
    if len(test_x) != 0:
        print("evaluating svm")
        test_scores = clf.predict_proba(test_x)
        print('accuracy for svm = ', str(np.mean(np.argmax(test_scores, axis=1) == test_y)))
    else:
        test_scores = []
    train_scores = clf.predict_proba(train_x)
    return train_scores, test_scores

def svm_scores_exists(dataset, network_name="inception",
                      alternative_data_dir="."):
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    return os.path.exists(svm_train_path) and os.path.exists(svm_test_path)

def get_svm_scores(transfer_values_train, y_train, transfer_values_test,
                   y_test, dataset, network_name="inception",
                   alternative_data_dir="."):
    
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    if not os.path.exists(svm_train_path) or not os.path.exists(svm_test_path):
        train_scores, test_scores = transfer_values_svm_scores(transfer_values_train, y_train, transfer_values_test, y_test)
        with open(svm_train_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)

        with open(svm_test_path, 'wb') as file_pi:
            pickle.dump(test_scores, file_pi)
    else:
        with open(svm_train_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)

        with open(svm_test_path, 'rb') as file_pi:
            test_scores = pickle.load(file_pi)
    return train_scores, test_scores


def rank_data_according_to_score(train_scores, y_train, reverse=False, random=False):
    train_size, _ = train_scores.shape
    hardness_score = train_scores[list(range(train_size)), y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)
    return res

def get_transfer_values_clap(dataset):
    """
    Loads pre-computed CLAP embeddings for the ESC50 dataset.
    
    Args:
        dataset: An instance of the ESC50 dataset class
        
    Returns:
        A tuple of (transfer_values_train, transfer_values_test) with CLAP embeddings
    """
    data_dir = r'./data/esc50_embeddings/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        from datasets import load_dataset
        embeddings_dataset = load_dataset("renumics/esc50-clap2023-results")
        # Save the dataset to the specified directory
        embeddings_dataset.save_to_disk(data_dir)
    else:
        from datasets import load_from_disk
        embeddings_dataset = load_from_disk(data_dir)

    embeddings_dataset = embeddings_dataset["train"]
    embeddings_dataset = [e["audio_embedding"] for e in embeddings_dataset]
    # Convert to numpy array
    embeddings_dataset = np.array(embeddings_dataset)
    # Split into train and test sets
    train_size = len(dataset.x_train)
    transfer_values_train = embeddings_dataset[:train_size]
    transfer_values_test = embeddings_dataset[train_size:]
    
    return transfer_values_train, transfer_values_test


if __name__ == "__main__":
    # Example usage
    from local_datasets.esc50 import ESC50
    esc50_dataset = ESC50(normalize=False)
    
    # Get transfer values using CLAP model
    transfer_values_train, transfer_values_test = get_transfer_values_clap(esc50_dataset)
    
    # Get SVM scores
    train_scores, test_scores = get_svm_scores(transfer_values_train, esc50_dataset.y_train,
                                               transfer_values_test, esc50_dataset.y_test,
                                               esc50_dataset, network_name="clap")
    
    print("Train scores shape:", train_scores.shape)
    print("Test scores shape:", test_scores.shape)
    print(f"Scores for first 5 training samples: {train_scores[:5]}")
    print(f"Scores for first 5 test samples: {test_scores[:5]}")
