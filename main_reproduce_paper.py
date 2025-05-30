#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:53:30 2019

@author: guy.hacohen
"""

from argparse import Namespace
from main_train_networks import run_expriment
import tensorflow as tf
print("Devices disponibles :")
print(tf.config.list_physical_devices())



def curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

    
def vanilla_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.035,
                     lr_decay_rate=1.8,
                     minimal_lr=1e-4,
                     lr_batch_size=600,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def anti_curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=200,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def random_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.3,
                     minimal_lr=1e-4,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def vanilla_cifar10_st_vgg(repeats, output_path=""):
    args = Namespace(dataset="cifar10",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def curriculum_cifar10_st_vgg(repeats, output_path=""):
    args = Namespace(dataset="cifar10",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def vanilla_cifar100_st_vgg(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def curriculum_cifar100_st_vgg(repeats, output_path=""):
    args = Namespace(dataset="cifar100",
                     model='stVGG',
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def curriculum_esc50_cnn(repeats, output_path="", lr=0.1, optimizer="sgd"):
    """
    Run curriculum learning experiment on ESC50 dataset with LSTM model
    using CLAP embeddings for transfer scoring
    """
    args = Namespace(dataset="esc50",
                     model='esc50',
                     output_path=output_path,
                     verbose=True,
                     optimizer=optimizer,
                     curriculum="curriculum",
                     batch_size=32,
                     num_epochs=120,
                     learning_rate=lr,
                     lr_decay_rate=1.2,
                     minimal_lr=1e-5,
                     lr_batch_size=300,
                     batch_increase=50,
                     increase_amount=1.5,
                     starting_percent=0.1,  # ESC-50 is smaller, start with more data
                     order="clap",  # Use CLAP embeddings for scoring
                     test_each=25,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def vanilla_esc50_cnn(repeats, output_path="", lr=0.1, optimizer="sgd"):
    """
    Run vanilla (no curriculum) experiment on ESC50 dataset with LSTM model
    """
    args = Namespace(dataset="esc50",
                     model='esc50',
                     output_path=output_path,
                     verbose=True,
                     optimizer=optimizer,
                     curriculum="vanilla",
                     batch_size=32,
                     num_epochs=120,
                     learning_rate=lr,
                     lr_decay_rate=1.2,
                     minimal_lr=1e-5,
                     lr_batch_size=300,
                     batch_increase=50,
                     increase_amount=1.5,
                     starting_percent=0.1,
                     order="clap",  # Still need this for comparison
                     test_each=25,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def anti_curriculum_esc50_cnn(repeats, output_path="", lr=0.1, optimizer="sgd"):
    """
    Run anti-curriculum learning experiment on ESC50 dataset with LSTM model
    using CLAP embeddings for transfer scoring (in reverse order)
    """
    args = Namespace(dataset="esc50",
                     model='esc50',
                     output_path=output_path,
                     verbose=True,
                     optimizer=optimizer,
                     curriculum="anti",
                     batch_size=32,
                     num_epochs=120,
                     learning_rate=lr,
                     lr_decay_rate=1.2,
                     minimal_lr=1e-5,
                     lr_batch_size=300,
                     batch_increase=50,
                     increase_amount=1.5,
                     starting_percent=0.1,
                     order="clap",
                     test_each=25,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


if __name__ == "__main__":
    
    output_path = "results/test_vanilla_cifar10"
    num_repeats = 1
    
#    import datasets.cifar100
#    
#    dataset = datasets.cifar100.Cifar100(False)
    
    
    #case 2 & 3
#    vanilla_cifar10_st_vgg(num_repeats, output_path=output_path)
#    curriculum_cifar10_st_vgg(num_repeats, output_path=output_path)
#    vanilla_cifar100_st_vgg(num_repeats, output_path=output_path)
    # curriculum_cifar100_st_vgg(num_repeats, output_path=output_path)
    
    # case 1
#    curriculum_small_mammals(num_repeats, output_path=output_path)
#    vanilla_small_mammals(num_repeats, output_path=output_path)
#    anti_curriculum_small_mammals(num_repeats, output_path=output_path)
#    random_small_mammals(num_repeats, output_path=output_path)
    output_path = "results/esc50_cnn"
    optimizers = ["sgd", "adam"]
    
    # ESC50 experiments
    lr = 0.1  # Default learning rate
    for optimizer in optimizers:
        for i in range(10):
            anti_curriculum_esc50_cnn(num_repeats, output_path=output_path + f"_anti_sgd_run_{i+1}", lr=lr, optimizer=optimizer)
            curriculum_esc50_cnn(num_repeats, output_path=output_path + f"_curriculum_sgd_run_{i+1}", lr=lr, optimizer=optimizer)
            vanilla_esc50_cnn(num_repeats, output_path=output_path + f"_vanilla_sgd_run_{i+1}", lr=lr, optimizer=optimizer)