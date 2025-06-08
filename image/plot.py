import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

#ADAM_MNIST/

#=======================================plot loss function training======================================================#
df_summary_vanilla = pd.read_csv('results_vanilla/summary_batch_metrics.csv')
df_summary_challenger = pd.read_csv('results_challenger/summary_batch_metrics.csv')
df_summary_curriculum = pd.read_csv('results_curriculum/summary_batch_metrics.csv')
df_summary_anti = pd.read_csv('results_anti/summary_batch_metrics.csv')

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

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Batch', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.grid(True)
plt.legend(loc='upper right', fontsize=24)
plt.tight_layout()
plt.show()

#=======================================plot validation accuracy======================================================#

df_val_vanilla = pd.read_csv('results_vanilla/periodic_validation_summary.csv')
df_val_challenger = pd.read_csv('results_challenger/periodic_validation_summary.csv')
df_val_curriculum = pd.read_csv('results_curriculum/periodic_validation_summary.csv')
df_val_anti = pd.read_csv('results_anti/periodic_validation_summary.csv')

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

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Batch', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.grid(True)
plt.legend(loc='lower right', fontsize=24)
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



plt.figure(figsize=(7, 7)) 

ax = plt.gca()
ax.set_facecolor('lightgrey') 
ax.spines['top'].set_color('white')  
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')

bars = plt.bar(labels, means, yerr=stds, color=colors, alpha=0.9, width=0.8, capsize=5, error_kw={'elinewidth':1.5})

for bar, mean in zip(bars, means):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=22)

plt.xticks(labels, fontsize=22, rotation=35)
plt.yticks(np.arange(0.9625, 0.9826, 0.005), fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.ylim(0.9625, 0.9825)
plt.grid(axis='y', linestyle='--', alpha=0.5, color='white')
plt.tight_layout()
plt.show()



#==================================Hypothese test=================================================#
# Assuming the column is named 'test_accuracy' and first 10 rows are runs
vanilla_accuracies = df_test_vanilla['test_accuracy'].iloc[:10].values
curriculum_accuracies = df_test_curriculum['test_accuracy'].iloc[:10].values
anti_accuracies = df_test_anti['test_accuracy'].iloc[:10].values
challenger_accuracies = df_test_challenger['test_accuracy'].iloc[:10].values

# Paired t-test: H0 = curriculum_mean <= vanilla_mean; H1 = curriculum_mean > vanilla_mean
t_stat, p_two_sided = ttest_rel(challenger_accuracies, vanilla_accuracies)

# One-sided p-value for curriculum > vanilla
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

print(f"t-statistic: {t_stat:.4f}")
print(f"One-sided p-value (curriculum > vanilla): {p_one_sided:.4f}")

alpha = 0.05
if p_one_sided < alpha:
    print("Reject H0: Curriculum model is significantly better than Vanilla")
else:
    print("Fail to reject H0: No significant difference")






