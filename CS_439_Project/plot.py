import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#=======================================plot loss function training======================================================#
df_summary_vanilla = pd.read_csv('results_vanilla\summary_batch_metrics.csv')
df_summary_challenger = pd.read_csv('results_challenger\summary_batch_metrics.csv')
df_summary_curriculum = pd.read_csv('results_curriculum\summary_batch_metrics.csv')
df_summary_anti = pd.read_csv('results_anti\summary_batch_metrics.csv')

# Create figure
plt.figure(figsize=(12, 5))

# Plot Vanilla with std shading
plt.plot(df_summary_vanilla['batch'], df_summary_vanilla['mean_loss'], label='Vanilla', color='blue')
plt.fill_between(
    df_summary_vanilla['batch'],
    df_summary_vanilla['mean_loss'] - df_summary_vanilla['std_loss'],
    df_summary_vanilla['mean_loss'] + df_summary_vanilla['std_loss'],
    color='blue',
    alpha=0.1,
    label='_nolegend_'  # Prevent duplicate legend entry
)

# Plot Challenger with std shading
plt.plot(df_summary_challenger['batch'], df_summary_challenger['mean_loss'], label='Challenger', color='green')
plt.fill_between(
    df_summary_challenger['batch'],
    df_summary_challenger['mean_loss'] - df_summary_challenger['std_loss'],
    df_summary_challenger['mean_loss'] + df_summary_challenger['std_loss'],
    color='green',
    alpha=0.1,
    label='_nolegend_'
)

# Curriculum
plt.plot(df_summary_curriculum['batch'], df_summary_curriculum['mean_loss'], label='Curriculum', color='orange')
plt.fill_between(
    df_summary_curriculum['batch'],
    df_summary_curriculum['mean_loss'] - df_summary_curriculum['std_loss'],
    df_summary_curriculum['mean_loss'] + df_summary_curriculum['std_loss'],
    color='orange', alpha=0.1, label='_nolegend_'
)

# Anti
plt.plot(df_summary_anti['batch'], df_summary_anti['mean_loss'], label='Anti-Curriculum', color='red')
plt.fill_between(
    df_summary_anti['batch'],
    df_summary_anti['mean_loss'] - df_summary_anti['std_loss'],
    df_summary_anti['mean_loss'] + df_summary_anti['std_loss'],
    color='red', alpha=0.1, label='_nolegend_'
)

# Formatting
plt.title('Training Loss over Batches with Std Deviation')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#=======================================plot validation accuracy======================================================#

df_val_vanilla = pd.read_csv('results_vanilla\periodic_validation_summary.csv')
df_val_challenger = pd.read_csv('results_challenger\periodic_validation_summary.csv')
df_val_curriculum = pd.read_csv('results_curriculum\periodic_validation_summary.csv')
df_val_anti = pd.read_csv('results_anti\periodic_validation_summary.csv')

# Create figure
plt.figure(figsize=(12, 5))

# Plot Vanilla with std shading
plt.plot(df_val_vanilla['batch'], df_val_vanilla['mean_val_accuracy'], label='Vanilla', color='blue')
plt.fill_between(
    df_val_vanilla['batch'],
    df_val_vanilla['mean_val_accuracy'] - df_val_vanilla['std_val_accuracy'],
    df_val_vanilla['mean_val_accuracy'] + df_val_vanilla['std_val_accuracy'],
    color='blue',
    alpha=0.1,
    label='_nolegend_'  # Prevent duplicate legend entry
)

# Plot Challenger with std shading
plt.plot(df_val_challenger['batch'], df_val_challenger['mean_val_accuracy'], label='Challenger', color='green')
plt.fill_between(
    df_val_challenger['batch'],
    df_val_challenger['mean_val_accuracy'] - df_val_challenger['std_val_accuracy'],
    df_val_challenger['mean_val_accuracy'] + df_val_challenger['std_val_accuracy'],
    color='green',
    alpha=0.1,
    label='_nolegend_'
)

# Curriculum
plt.plot(df_val_curriculum['batch'], df_val_curriculum['mean_val_accuracy'], label='Curriculum', color='orange')
plt.fill_between(
    df_val_curriculum['batch'],
    df_val_curriculum['mean_val_accuracy'] - df_val_curriculum['std_val_accuracy'],
    df_val_curriculum['mean_val_accuracy'] + df_val_curriculum['std_val_accuracy'],
    color='orange', alpha=0.1, label='_nolegend_'
)

# Anti-Curriculum
plt.plot(df_val_anti['batch'], df_val_anti['mean_val_accuracy'], label='Anti-Curriculum', color='red')
plt.fill_between(
    df_val_anti['batch'],
    df_val_anti['mean_val_accuracy'] - df_val_anti['std_val_accuracy'],
    df_val_anti['mean_val_accuracy'] + df_val_anti['std_val_accuracy'],
    color='red', alpha=0.1, label='_nolegend_'
)

# Formatting
plt.title('Validation Accuracy over Batches with Std Deviation')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#=======================================Histogram test accuracy======================================================#
df_test_vanilla = pd.read_csv('results_vanilla/test_results_summary.csv')
df_test_challenger = pd.read_csv('results_challenger/test_results_summary.csv')
df_test_curriculum = pd.read_csv('results_curriculum/test_results_summary.csv')
df_test_anti = pd.read_csv('results_anti/test_results_summary.csv')

# Get second-to-last test_accuracy value for all datasets
vanilla_mean = df_test_vanilla['test_accuracy'].iloc[-2]
vanilla_std = df_test_vanilla['test_accuracy'].iloc[-1]

challenger_mean = df_test_challenger['test_accuracy'].iloc[-2]
challenger_std = df_test_challenger['test_accuracy'].iloc[-1]

curriculum_mean = df_test_curriculum['test_accuracy'].iloc[-2]
curriculum_std = df_test_curriculum['test_accuracy'].iloc[-1]

anti_mean = df_test_anti['test_accuracy'].iloc[-2]
anti_std = df_test_anti['test_accuracy'].iloc[-1]

labels = ['Vanilla', 'Challenger', 'Curriculum', 'Anti']
means = [vanilla_mean, challenger_mean, curriculum_mean, anti_mean]
stds = [vanilla_std, challenger_std, curriculum_std, anti_std]
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, means, yerr=stds, color=colors, alpha=0.7, width=0.8, capsize=5, error_kw={'elinewidth':1.5})

# Add mean values on top of bars
for bar, mean in zip(bars, means):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f'{mean:.3f}', ha='center', va='bottom')

plt.xticks(labels)
plt.title('Test Accuracy with Standard Deviation')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()










#histogram accuracy test