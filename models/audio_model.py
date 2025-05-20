"""
AudioModel class for ESC-50 and other audio datasets.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

class AudioModel:
    """
    Model for audio classification tasks.
    """
    
    def __init__(self):
        """Initialize the AudioModel."""
        pass
        
    def build_classifier_model(self, dataset):
        """
        Build and return a CNN model for audio classification.
        
        Parameters
        ----------
        dataset : object
            The dataset object with necessary information about input shape
            and number of classes.
            
        Returns
        -------
        model : tf.keras.Model
            The compiled model ready for training.
        """
        # Get shapes from dataset
        if hasattr(dataset, 'x_train') and dataset.x_train is not None:
            input_shape = dataset.x_train.shape[1:]
            num_classes = dataset.n_classes
        else:
            # Default values if dataset doesn't have the expected attributes
            input_shape = (128, 128, 1)  # Default mel spectrogram shape
            num_classes = 50  # ESC-50 has 50 classes
        
        # Create model
        model = models.Sequential()
        
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        return model
