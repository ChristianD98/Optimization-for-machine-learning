import pandas as pd
import matplotlib.pyplot as plt

# Charger les fichiers CSV
train_df = pd.read_csv("results/test_run2_train_history.csv")
val_df = pd.read_csv("results/test_run2_val_history.csv")

# Créer le graphique
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
plt.show()