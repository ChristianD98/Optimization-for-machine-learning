import numpy as np
from keras.callbacks import Callback
from keras.utils import Sequence


###############################################################################################

#IMPLEMENT THE OMEGA FROM WICH WE RANDOMLY EXCTRACT OUR DATA

class PacingGenerator(Sequence):
    def __init__(self, x_data, y_data, stddevs, batch_size=64, mode='vanilla', curriculum_epochs=10, starting_fraction=0.05, inc=1.9, step_length=100):
        """
        Args:
            x_data: Full training images (sorted or unsorted).
            y_data: Corresponding labels.
            stddevs: Stddevs associated with images.
            batch_size: Batch size.
            mode: 'vanilla' or 'curriculum'.
            curriculum_epochs: Number of epochs over which to gradually increase difficulty.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.stddevs = stddevs
        self.batch_size = batch_size
        self.mode = mode
        self.curriculum_epochs = curriculum_epochs
        self.sorted_indices = np.argsort(stddevs)
        self.epoch = 0
        self.on_epoch_end()
        self.starting_fraction = starting_fraction
        self.inc = inc
        self.step_length = step_length


    def __len__(self):
        return len(self.x_data) // self.batch_size

    def __getitem__(self, idx):
        if self.mode == 'vanilla':
            indices = np.random.choice(len(self.x_data), self.batch_size, replace=True)
            
        elif self.mode == 'curriculum':
            # Total number of training steps so far
            current_step = self.epoch * (len(self.x_data) // self.batch_size)
            step_idx = current_step // self.step_length

            # Compute pace(i)
            pace_fraction = min(1.0, self.starting_fraction * (self.inc ** step_idx))
            max_difficulty_index = int(len(self.sorted_indices) * pace_fraction)

            current_pool = self.sorted_indices[:max_difficulty_index]
            indices = np.random.choice(current_pool, self.batch_size, replace=True)
            
        elif self.mode == 'anti':
            # Total number of training steps so far
            current_step = self.epoch * (len(self.x_data) // self.batch_size)
            step_idx = current_step // self.step_length

            # Compute pace(i)
            anti_sorted_indices = np.flip(self.sorted_indices, 0)
            pace_fraction = min(1.0, self.starting_fraction * (self.inc ** step_idx))
            max_difficulty_index = int(len(self.sorted_indices) * pace_fraction)

            current_pool = np.flip(anti_sorted_indices[:max_difficulty_index], 0)
            indices = np.random.choice(current_pool, self.batch_size, replace=True)
            
        elif self.mode == 'challenger':
            # Total number of batches seen so far
            total_batches = self.epoch * (len(self.x_data) // self.batch_size) + idx

            # Define challenger pacing parameters
            batches_to_increase = self.step_length
            increase_amount = self.inc
            pace_fraction = min(1.0, self.starting_fraction * (increase_amount ** (total_batches // batches_to_increase)))

            # Define dynamic window: if past 1/4 of interval, reset lower bound to 0
            if (total_batches % batches_to_increase) > (batches_to_increase // 4):
                lower_fraction = 0.0
            else:
                lower_fraction = max(0.0, pace_fraction / increase_amount)

            lower_idx = int(len(self.sorted_indices) * lower_fraction)
            upper_idx = int(len(self.sorted_indices) * pace_fraction)

            current_pool = self.sorted_indices[lower_idx:upper_idx]
            if len(current_pool) == 0:
                current_pool = self.sorted_indices[:int(len(self.sorted_indices) * self.starting_fraction)]
            indices = np.random.choice(current_pool, self.batch_size, replace=True)
        
        else:
            raise ValueError("Unsupported mode")
        
        return self.x_data[indices], self.y_data[indices]

    def on_epoch_end(self):
        self.epoch += 1


class BatchMetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_logs = []
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['batch'] = self.batch_count
        self.batch_logs.append(logs.copy())
        self.batch_count += 1
        
# Custom callback for periodic validation
class PeriodicValidationCallback(Callback):
    def __init__(self, validation_data, interval=100):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.interval = interval
        self.validation_log = []
        self.batch_counter = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.interval == 0:
            loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)
            self.validation_log.append({
                'batch': self.batch_counter,
                'val_accuracy': accuracy
            })
