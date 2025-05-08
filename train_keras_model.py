#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:33:53 2018

@author: guy.hacohen
"""
import keras
import numpy as np
import keras.backend as K
import time

def compile_model(model, initial_lr=1e-3, loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'], momentum=0.0):
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(initial_lr, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=0.0,
                                          amsgrad=False)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(initial_lr, momentum=momentum)
    else:
        print("optimizer not supported")
        raise ValueError
    
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


def basic_data_function(x_train, y_train, batch, history, model):
    return x_train, y_train

def basic_lr_scheduler(initial_lr, batch, history):
    return initial_lr


def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]

def generate_curriculum_batch_easy_hard_mix(x, y, batch_size, batch_index, num_batches):
    total_size = x.shape[0]
    half = total_size // 2

    easy_x = x[:half]
    easy_y = y[:half]
    hard_x = x[half:]
    hard_y = y[half:]

    # Calcul du pourcentage easy (linéaire de 0.95 → 0.05)
    easy_pct = max(0.05, 0.95 - 0.9 * (batch_index / num_batches))
    hard_pct = 1 - easy_pct

    num_easy = int(round(batch_size * easy_pct))
    num_hard = batch_size - num_easy

    easy_idx = np.random.choice(len(easy_x), num_easy, replace=False)
    hard_idx = np.random.choice(len(hard_x), num_hard, replace=False)

    batch_x = np.concatenate((easy_x[easy_idx], hard_x[hard_idx]), axis=0)
    batch_y = np.concatenate((easy_y[easy_idx], hard_y[hard_idx]), axis=0)

    # Shuffle le batch
    perm = np.random.permutation(batch_size)
    pacing_info = {
        "batch": batch_index,
        "easy_pct": easy_pct,
        "hard_pct": hard_pct,
        "num_easy": num_easy,
        "num_hard": num_hard
    }

    return batch_x[perm], batch_y[perm], pacing_info

def train_model_batches(model, dataset, num_batches, batch_size=100,
                        test_each=50, batch_strategy="random", initial_lr=1e-3,
                     
   lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy',
                        data_function=basic_data_function,
                        verbose=False):
    
    x_train = dataset.x_train
    y_train = dataset.y_train_labels
    x_test = dataset.x_test
    y_test = dataset.y_test_labels

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "batch_num": [], "data_size": []}
    pacing_record = []
    start_time = time.time()
    for batch in range(num_batches):
        cur_x, cur_y = data_function(x_train, y_train, batch, history, model)
        cur_lr = lr_scheduler(initial_lr, batch, history)
        model.optimizer.learning_rate.assign(cur_lr)

        if batch_strategy == "random":
            batch_x, batch_y = generate_random_batch(cur_x, cur_y, batch_size)
            pacing_record = None
        elif batch_strategy == "curriculum_easy_hard":
            batch_x, batch_y, pacing_info = generate_curriculum_batch_easy_hard_mix(cur_x, cur_y, batch_size, batch, num_batches)
            pacing_record.append(pacing_info)
        else:
            raise ValueError(f"Unknown batch_strategy: {batch_strategy}")
        
        cur_loss, cur_accuracy = model.train_on_batch(batch_x, batch_y)
        history["loss"].append(cur_loss)
        history["acc"].append(cur_accuracy)
        history["data_size"].append(cur_x.shape[0])
        if test_each is not None and (batch+1) % test_each == 0:
            cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
            history["val_loss"].append(cur_val_loss)
            history["val_acc"].append(cur_val_acc)
            history["batch_num"].append(batch)
            if verbose:
                print("val accuracy:", cur_val_acc)
        if verbose and (batch+1) % 5 == 0:
            print("batch: " + str(batch+1) + r"/" + str(num_batches))
            print("last lr used: " + str(cur_lr))
            print("data_size: " + str(cur_x.shape[0]))
            print("loss: " + str(cur_loss))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    if pacing_record:
        history["pacing"] = pacing_record

    return history