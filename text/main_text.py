import os
import pandas as pd
import numpy as np
from data.text_data import load_imdb_labels, vocab_size
from text_model import IMDB_Model
from scoring_text import get_infrequency_scores, get_length_scores, get_combined_scores
from pacing import PacingGenerator, BatchMetricsLogger, PeriodicValidationCallback
import tensorflow as tf
import pickle

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ===========================
n_runs = 5
mode = 'curriculum'  # 'vanilla', 'curriculum', 'anti', 'challenger'
output_dir = f"./results_{mode}"
data_type = 'text'
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(output_dir, data_type)
os.makedirs(data_dir, exist_ok=True)

with open("Transfer_Learning/text/teacher_embeddings.pkl", "rb") as f:
    x_train_embeddings = pickle.load(f)
with open("Transfer_Learning/text/test_embeddings.pkl", "rb") as f:
    x_test_embeddings = pickle.load(f)
with open("Transfer_Learning/text/difficulty_scores.pkl", "rb") as f:
    difficulty_scores = pickle.load(f)

label_namespace = load_imdb_labels(vocab_size=vocab_size, num_classes=2)

all_batch_logs = []
test_results = []
all_validation_logs = []

for run in range(n_runs):
    set_seed(run)
    print(f"\n===== Starting run {run+1}/{n_runs} (seed={run})=====")

    batch_logger = BatchMetricsLogger()
    periodic_val_logger = PeriodicValidationCallback(
        validation_data=(x_test_embeddings, label_namespace.y_test), interval=100
    )

    omega_generator = PacingGenerator(
        x_data=x_train_embeddings,
        y_data=label_namespace.y_train,
        # ***** Choose between these difficulty scores *****
        # 1) difficulty_scores
        # 2) get_infrequency_scores(x_data = x_train_embeddings)
        # 3) get_length_scores(x_data = x_train_embeddings)
        # 4) get_combined_scores(x_data = x_train_embeddings, alpha=0.5),
        stddevs=difficulty_scores, 
        batch_size=64,
        mode=mode,
        curriculum_epochs=3,
        starting_fraction=0.05,
        inc=1.9,
        step_length=100
    )

    model = IMDB_Model().build_mlp_model(embedding_dim=x_train_embeddings.shape[1])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # Choose between 'adam' or 'sgd'

    model.fit(
        omega_generator,
        epochs=3,
        callbacks=[batch_logger, periodic_val_logger],
        verbose=1
    )

    # Store batch logs
    df = pd.DataFrame(batch_logger.batch_logs)
    df['run'] = run
    all_batch_logs.append(df)

    # Store periodic validation logs
    val_df = pd.DataFrame(periodic_val_logger.validation_log)
    val_df['run'] = run
    all_validation_logs.append(val_df)

    # Store test set evaluation
    loss, accuracy = model.evaluate(x_test_embeddings, label_namespace.y_test)
    test_results.append({'run': run + 1, 'test_loss': loss, 'test_accuracy': accuracy})


# ====================================== Data collection ================================ #
combined_df = pd.concat(all_batch_logs, ignore_index=True)
summary_df = combined_df.groupby('batch').agg({
    'loss': ['mean', 'std'],
    'accuracy': ['mean', 'std']
}).reset_index()
summary_df.columns = ['batch', 'mean_loss', 'std_loss', 'mean_accuracy', 'std_accuracy']
summary_df.to_csv(f"{output_dir}/{data_type}/summary_batch_metrics.csv", index=False)

results_df = pd.DataFrame(test_results)
mean_row = results_df.mean(numeric_only=True)
std_row = results_df.std(numeric_only=True)
results_df = pd.concat([
    results_df,
    pd.DataFrame([mean_row.rename('mean'), std_row.rename('std')])
])
results_df.to_csv(f"{output_dir}/{data_type}/test_results_summary.csv", index=False)

val_combined_df = pd.concat(all_validation_logs, ignore_index=True)
val_summary_df = val_combined_df.groupby('batch')['val_accuracy'].agg(['mean', 'std']).reset_index()
val_summary_df.columns = ['batch', 'mean_val_accuracy', 'std_val_accuracy']
val_summary_df.to_csv(f"{output_dir}/{data_type}/periodic_validation_summary.csv", index=False)

print(f"✅ Per-batch metrics summary saved to '{output_dir}/{data_type}/summary_batch_metrics.csv'")
print(f"✅ Test results with mean and std saved to '{output_dir}/{data_type}/test_results_summary.csv'")
print(f"✅ Periodic validation accuracy summary saved to '{output_dir}/{data_type}/periodic_validation_summary.csv'")
