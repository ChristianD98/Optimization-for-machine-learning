from sentence_transformers import SentenceTransformer
from keras.datasets import imdb
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Récupérer le vocabulaire
word_index = imdb.get_word_index()
reverse_word_index = {value+3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review if i > 2])


(train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=10000)

train_texts = [decode_review(x) for x in train_X]
test_texts = [decode_review(x) for x in test_X]

model = SentenceTransformer('all-MiniLM-L6-v2')

train_embeddings = model.encode(train_texts, batch_size=64, show_progress_bar=True)
test_embeddings = model.encode(test_texts, batch_size=64, show_progress_bar=True)

scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

clf = SVC(probability=True)
print("Training SVM on train_embeddings...")
clf.fit(train_embeddings_scaled, train_y)

test_preds = clf.predict(test_embeddings_scaled)
acc = accuracy_score(test_y, test_preds)
print(f"Test accuracy of the SVM: {acc:.4f}")

train_probas = clf.predict_proba(train_embeddings_scaled)
train_confidences = train_probas[np.arange(len(train_y)), train_y]

difficulty_scores = 1.0 - train_confidences

sorted_indices = np.argsort(difficulty_scores)

for i in sorted_indices[:5]:
    print(f"\nReview {i} (difficulty {difficulty_scores[i]:.4f}):")
    print(train_texts[i])

import pickle

with open("train_texts.pkl", "wb") as f:
    pickle.dump(train_texts, f)

with open("test_texts.pkl", "wb") as f:
    pickle.dump(test_texts, f)

with open("teacher_embeddings.pkl", "wb") as f:
    pickle.dump(train_embeddings_scaled, f)

with open("test_embeddings.pkl", "wb") as f:
    pickle.dump(test_embeddings_scaled, f)

with open("difficulty_scores.pkl", "wb") as f:
    pickle.dump(difficulty_scores, f)
