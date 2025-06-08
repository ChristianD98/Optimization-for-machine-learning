from argparse import Namespace
from main_train_networks import run_expriment
import tensorflow as tf
print("Devices available :")
print(tf.config.list_physical_devices())


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


def curriculum_mixed_esc50_cnn(repeats, output_path="", lr=0.1, optimizer="sgd"):
    """
    Run mixed curriculum learning experiment on ESC50 dataset with CNN model
    using CLAP embeddings for transfer scoring
    """
    args = Namespace(dataset="esc50",
                     model='esc50',
                     output_path=output_path,
                     verbose=True,
                     optimizer=optimizer,
                     curriculum="curriculum_mixed",
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


if __name__ == "__main__":
    num_repeats = 1
    output_path = "results/esc50_cnn/"
    optimizers = ["sgd", "adam"]
    
    # ESC50 experiments
    lr = 0.1  # Default learning rate
    for optimizer in optimizers:
        for i in range(3):
            curriculum_mixed_esc50_cnn(num_repeats, output_path=output_path + f"{optimizer}_curriculum_mixed_{optimizer}_run_{i+1}", lr=lr, optimizer=optimizer)
            anti_curriculum_esc50_cnn(num_repeats, output_path=output_path + f"{optimizer}_anti_{optimizer}_run_{i+1}", lr=lr, optimizer=optimizer)
            curriculum_esc50_cnn(num_repeats, output_path=output_path + f"{optimizer}_curriculum_{optimizer}_run_{i+1}", lr=lr, optimizer=optimizer)
            vanilla_esc50_cnn(num_repeats, output_path=output_path + f"{optimizer}_vanilla_{optimizer}_run_{i+1}", lr=lr, optimizer=optimizer)