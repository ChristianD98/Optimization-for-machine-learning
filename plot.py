import pandas as pd
import matplotlib.pyplot as plt

# === Chargement des fichiers ===
train_df = pd.read_csv("results/test_custom_train_history.csv")
val_df = pd.read_csv("results/test_custom_val_history.csv")
pacing_df = pd.read_csv("results/test_custom_pacing2.csv")

# === Figure 1 : Training vs Validation Accuracy ===
plt.figure(figsize=(12, 6))

# Courbe d'entraînement
plt.plot(train_df["train_batch"], train_df["acc"], label="Training Accuracy")
if "std_acc" in train_df.columns and not train_df["std_acc"].isnull().all():
    plt.fill_between(train_df["train_batch"],
                     train_df["acc"] - train_df["std_acc"],
                     train_df["acc"] + train_df["std_acc"],
                     alpha=0.3, label="Train Std")

# Courbe de validation
plt.plot(val_df["batch_num"], val_df["val_acc"], label="Validation Accuracy")
if "std_val_acc" in val_df.columns and not val_df["std_val_acc"].isnull().all():
    plt.fill_between(val_df["batch_num"],
                     val_df["val_acc"] - val_df["std_val_acc"],
                     val_df["val_acc"] + val_df["std_val_acc"],
                     alpha=0.3, label="Val Std")

# Labels et légende
plt.xlabel("Batch Number")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Figure 2 : Easy vs Hard Sample Ratio par batch ===
plt.figure(figsize=(12, 5))
plt.plot(pacing_df["batch"], pacing_df["easy_pct"], label="Easy Sample %")
plt.plot(pacing_df["batch"], pacing_df["hard_pct"], label="Hard Sample %")
plt.xlabel("Batch Number")
plt.ylabel("Sample Percentage")
plt.title("Pacing Schedule: Easy vs Hard Sample Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
