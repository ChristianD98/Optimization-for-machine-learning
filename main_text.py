import os
import pandas as pd
import numpy as np
from data.text_data import load_imdb_dataset, get_word_index, load_glove_embedding_matrix, vocab_size, maxlen
from text_model import IMDB_Model
from scoring_text import get_combined_scores
from pacing import PacingGenerator, BatchMetricsLogger, PeriodicValidationCallback
import tensorflow as tf
import pickle

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ===========================
n_runs = 2
mode = 'vanilla'  # 'vanilla', 'curriculum', 'anti', 'challenger'
output_dir = f"./results_{mode}"
data_type = 'text' # 'text', 'img', 'sound'
scoring = "Transfer_Learning"
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(output_dir, data_type)
os.makedirs(data_dir, exist_ok=True)

all_batch_logs = []
test_results = []
all_validation_logs = []

for run in range(n_runs):
    set_seed(run)
    print(f"\n===== Starting run {run+1}/{n_runs} (seed={run})=====")

    dataset = load_imdb_dataset(vocab_size=vocab_size, maxlen=maxlen, num_classes=2)
    batch_logger = BatchMetricsLogger()
    periodic_val_logger = PeriodicValidationCallback(
        validation_data=(dataset.x_test, dataset.y_test), interval=100
    )

    if scoring == "Transfer_Learning":
        with open("difficulty_scores.pkl", "rb") as f:
            difficulty_scores = pickle.load(f)

    omega_generator = PacingGenerator(
        x_data=dataset.x_train,
        y_data=dataset.y_train,
        stddevs=difficulty_scores,
        batch_size=64,
        mode=mode,
        curriculum_epochs=10,
        starting_fraction=0.2,
        inc=1.3,
        step_length=500
    )

    word_index = get_word_index()
    glove_path = "embedding/glove.6B/glove.6B.200d.txt"
    embedding_matrix = load_glove_embedding_matrix(glove_path, word_index, vocab_size=vocab_size, embedding_dim=200)

    model = IMDB_Model().build_classifier_model(embedding_matrix=embedding_matrix)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        omega_generator,
        epochs=10,
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
    loss, accuracy = model.evaluate(dataset.x_test, dataset.y_test)
    test_results.append({'run': run + 1, 'test_loss': loss, 'test_accuracy': accuracy})





# ====================================== Data collection ================================ #
# Per-batch training metric summary
combined_df = pd.concat(all_batch_logs, ignore_index=True)
summary_df = combined_df.groupby('batch').agg({
    'loss': ['mean', 'std'],
    'accuracy': ['mean', 'std']
}).reset_index()
summary_df.columns = ['batch', 'mean_loss', 'std_loss', 'mean_accuracy', 'std_accuracy']
summary_df.to_csv(f"{output_dir}/{data_type}/summary_batch_metrics.csv", index=False)

# Test set results (mean + std)
results_df = pd.DataFrame(test_results)
mean_row = results_df.mean(numeric_only=True)
std_row = results_df.std(numeric_only=True)
results_df = pd.concat([
    results_df,
    pd.DataFrame([mean_row.rename('mean'), std_row.rename('std')])
])
results_df.to_csv(f"{output_dir}/{data_type}/test_results_summary.csv", index=False)

# Periodic validation accuracy summary
val_combined_df = pd.concat(all_validation_logs, ignore_index=True)
val_summary_df = val_combined_df.groupby('batch')['val_accuracy'].agg(['mean', 'std']).reset_index()
val_summary_df.columns = ['batch', 'mean_val_accuracy', 'std_val_accuracy']
val_summary_df.to_csv(f"{output_dir}/{data_type}/periodic_validation_summary.csv", index=False)

# Final confirmation
print(f"✅ Per-batch metrics summary saved to '{output_dir}/{data_type}/summary_batch_metrics.csv'")
print(f"✅ Test results with mean and std saved to '{output_dir}/{data_type}/test_results_summary.csv'")
print(f"✅ Periodic validation accuracy summary saved to '{output_dir}/{data_type}/periodic_validation_summary.csv'")
