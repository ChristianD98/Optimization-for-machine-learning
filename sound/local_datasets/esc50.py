#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
try:
    import local_datasets.Dataset as Dataset
except ImportError:
    import Dataset
try:
    from local_datasets.Dataset import one_hot_encoded
except ImportError:
    from Dataset import one_hot_encoded

from huggingface_hub import hf_hub_download, snapshot_download
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm

class ESC50(Dataset.Dataset):
    def __init__(self, normalize=True):
        self.name = 'esc50'
        self.data_url = "https://huggingface.co/datasets/karoldvl/ESC-50/resolve/main/esc50.json"
        self.data_dir = "./data/esc50/"
        self.n_classes = 50
        self.dataset = None  # Placeholder for the dataset

        super(ESC50, self).__init__(normalize=normalize)


    def maybe_download(self):
        """
        Download the dataset if it does not exist.

        This method checks if the dataset is already downloaded and available locally.
        If not, it downloads the dataset from the specified URL and saves it in the data directory.
        If the dataset is already downloaded, it does nothing.

        :return: None
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        dataset = load_dataset("renumics/esc50")
        dataset.save_to_disk(self.data_dir)
        del dataset
        print(f"Dataset downloaded and saved to {self.data_dir}")

    def load_training_data(self):
        """
        Load the training data from the ESC-50 dataset.

        This method reads the ESC-50 dataset json files, extracts audio features and labels,
        and returns them as numpy arrays.

        :return: Tuple of (features, labels, one-hot encoded labels)
        """
        # check if data directory exists, if not download it
        if not self.dataset:
            if not os.path.exists(self.data_dir):
                self.maybe_download()
            else:
                # If the dataset is already downloaded, load it from the local directory
                print(f"Loading dataset from {self.data_dir}")
                dataset = load_dataset(self.data_dir)
            self.dataset = dataset
            del dataset

        
        train_data = self.dataset['train']
        # train is 4 folds
        x_train = []
        y_train = []
        for row in tqdm(train_data):
            if row["fold"] < 5:
                feat = self.to_mel_spectrogram(row['audio']['array'], sr=row['audio']['sampling_rate'], n_mels=128, fmax=8000)
                # add an axis for the channel dimension
                feat = np.expand_dims(feat, axis=-1)  # shape (n_mels, time_steps, 1)
                x_train.append(feat)
                y_train.append(row['label'])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)

        return x_train, y_train, y_train_labels
    
    def load_test_data(self):
        """
        Load the test data from the ESC-50 dataset.

        This method reads the ESC-50 dataset json files, extracts audio features and labels,
        and returns them as numpy arrays.

        :return: Tuple of (features, labels, one-hot encoded labels)
        """
        # check if data directory exists, if not download it
        if not self.dataset:
            if not os.path.exists(self.data_dir):
                self.maybe_download()
            else:
                # If the dataset is already downloaded, load it from the local directory
                print(f"Loading dataset from {self.data_dir}")
                dataset = load_dataset(self.data_dir)
            self.dataset = dataset
            del dataset

        test_data = self.dataset['train']
        
        x_test = []
        y_test = []
        for row in tqdm(test_data):
            if row["fold"] == 5:
                feat = self.to_mel_spectrogram(row['audio']['array'], sr=row['audio']['sampling_rate'], n_mels=128, fmax=8000)
                # add an axis for the channel dimension
                feat = np.expand_dims(feat, axis=-1)  # shape (n_mels, time_steps, 1)
                x_test.append(feat)
                y_test.append(row['label'])

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)

        return x_test, y_test, y_test_labels
    
    def normalize_dataset(self):
        pass

    def to_mel_spectrogram(self, audio, sr=16000, n_mels=128, fmax=8000):
        """
        Convert audio signal to mel spectrogram.

        :param audio: Audio signal as a numpy array.
        :param sr: Sample rate of the audio signal.
        :param n_mels: Number of mel bands to generate.
        :param fmax: Maximum frequency to consider in the mel scale.
        :return: Mel spectrogram as a numpy array.
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
        return librosa.power_to_db(mel_spectrogram, ref=np.max)
    


if __name__ == "__main__":
    esc50_dataset = ESC50(normalize=True)
    esc50_dataset.maybe_download()
    # x_train, y_train, y_train_labels = esc50_dataset.load_training_data()
    # x_test, y_test, y_test_labels = esc50_dataset.load_test_data()
    
    print(f"Training data shape: {esc50_dataset.x_train.shape}")
    print(f"Training labels shape: {esc50_dataset.y_train.shape}")
    print(f"One-hot encoded labels shape: {esc50_dataset.y_train_labels.shape}")
    print(f"="*20)
    print(f"Test data shape: {esc50_dataset.x_test.shape}")
    print(f"Test labels shape: {esc50_dataset.y_test.shape}")
    print(f"One-hot encoded test labels shape: {esc50_dataset.y_test_labels.shape}")
    print(f"Number of classes: {esc50_dataset.n_classes}")