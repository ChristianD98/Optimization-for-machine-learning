from sentence_transformers import SentenceTransformer
from keras.datasets import imdb
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Récupérer le vocabulaire
word_index = imdb.get_word_index()
reverse_word_index = {value+3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"

# Fonction de décodage
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review if i > 2])


(train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=10000)

# Pour commencer rapidement, on prend un sous-ensemble
train_texts = [decode_review(x) for x in train_X]
test_texts = [decode_review(x) for x in test_X]

model = SentenceTransformer('all-MiniLM-L6-v2')

train_embeddings = model.encode(train_texts, batch_size=64, show_progress_bar=True)
test_embeddings = model.encode(test_texts, batch_size=64, show_progress_bar=True)


# SVM avec sortie de probas
clf = SVC(probability=True)
print("🧠 Training SVM on train_embeddings...")
clf.fit(train_embeddings, train_y)

# Évaluer sur le test set (facultatif)
test_preds = clf.predict(test_embeddings)
acc = accuracy_score(test_y, test_preds)
print(f"✅ Test accuracy of the SVM: {acc:.4f}")

# Confiance du SVM sur la classe correcte
train_probas = clf.predict_proba(train_embeddings)  # shape (5000, 2)
train_confidences = train_probas[np.arange(len(train_y)), train_y]

# Score de difficulté = 1 - confiance
difficulty_scores = 1.0 - train_confidences

# Trier les indices du plus facile au plus difficile
sorted_indices = np.argsort(difficulty_scores)

# Exemple : top 5 textes les plus faciles
for i in sorted_indices[:5]:
    print(f"\nReview {i} (difficulty {difficulty_scores[i]:.4f}):")
    print(train_texts[i])

import pickle

with open("teacher_embeddings.pkl", "wb") as f:
    pickle.dump(train_embeddings, f)

with open("difficulty_scores.pkl", "wb") as f:
    pickle.dump(difficulty_scores, f)
