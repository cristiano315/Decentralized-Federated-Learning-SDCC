#File for the model definition, and training, to implement
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
# reduce words
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier

# testing
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentPyTorch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(SentimentPyTorch, self).__init__()
        # Il layer EmbeddingBag calcola la media degli embedding dei token nel tweet
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

SEED = 42

def tokenize(tweet_text):
    tknzr = TweetTokenizer()
    tweet_tokens = tknzr.tokenize(tweet_text)
    return tweet_tokens

def tweet_to_tensor(tokenized_tweet, word_to_idx):
    # Converte i token in indici, ignorando parole fuori vocabolario
    indices = [word_to_idx[w] for w in tokenized_tweet if w in word_to_idx]
    return torch.tensor(indices, dtype=torch.int64)

punct = {'💁', '[', '’', '~', '💪', '📚', '🏡', '-', '🐣', '🇺', '”', '̶', '\u200a', ';', '🍕', ' ', '!', '%', ',', '👇', '®', '🌈', '?', '🏽', '=', '💨', '✅', '✔', ')', '|', '‘', '\xa0', '🗽', '&', '🏼', '¿', '…', '🎓', '👉', '❌', '🎧', '👈', '🚂', '+', '🤖', '👎', '→', '¡', '🤔', '️', '👸', '@', '🇸', ':', '“', '•', '🏿', '🏻', '👀', '👏', '—', ']', '✓', '"', '\u200b', '🎤', '\n', '.', '(', '$', '❤', '⬇', '#', '👍', "'", '/', '*', '🏾', '–', '👿'}

def clean(text):
    temp_text = text.lower()
    temp_text = re.sub(r'https?://\S+', '', temp_text)
    temp_text = re.sub(r'\d+', '0', temp_text)
    temp_text = temp_text.replace('’', '\'')  # some single quotes are slanted, and we want to retain them
    for p in punct:
        temp_text = temp_text.replace(p, ' ')
    temp_text = re.sub(r'\s+', ' ', temp_text)
    cleaned_text = temp_text.strip()
    return cleaned_text

def evaluate(model, data):

    model.eval()

    acc = 0.0

    # calculate accuracy based on predictions
    ############### for student ################
    accuracy_list = []
    for i, sentence_tag in enumerate(data):
      sentence = [word_to_id[s[0]] for s in sentence_tag]
      sentence = torch.tensor(sentence, dtype=torch.long)
      sentence = sentence.to(device)
      targets = [tag_to_id[s[1]] for s in sentence_tag]
      targets = torch.tensor(targets, dtype=torch.long)
      targets = targets.to(device)

      tag_scores = model(sentence)

      _, indices = torch.max(tag_scores, 1)

      acc += torch.mean((targets == indices).float())

    acc = acc / len(data)
    accuracy_list.append(float(acc))
    score = acc.item()

    ############################################

    return score

# 1. Carica il file JSON con la libreria standard
file_path = "./all_data_niid_05_keep_3_train_9.json"
test_path = "./all_data_niid_05_keep_3_test_9.json"

# TRAIN TRAIN

with open(file_path, 'r') as f:
    raw_data = json.load(f)

# 2. Trasforma la struttura annidata in una lista piatta
flattened_data = []
for user in raw_data['users']:
    user_tweets = raw_data['user_data'][user]['x']
    user_labels = raw_data['user_data'][user]['y']
    
    for tweet, label in zip(user_tweets, user_labels):
        flattened_data.append({
            'user_id': user,
            'tweet_id': tweet[0],
            'date': tweet[1],
            'query': tweet[2],
            'author': tweet[3],
            'text': tweet[4],  # Il testo del tweet è all'indice 4
            'label': label
        })

# 3. Crea il DataFrame
data = pd.DataFrame(flattened_data)

print(f"Dataset caricato: {len(data)} tweet.")
print(data[['text', 'label']].head(30))
# Assemble lines: concatenate title and description
lines = data.apply(lambda row: row['text'], axis=1).tolist()


punct.remove(' ')  # keep spaces
punct.remove('#')  # keep hashtags
punct.remove('@')  # keep mentions
punct.remove('\'') # keep single quotes (in order to retain I'm, isn't, etc.)
tweet_text = data['text'].iloc[0]
print(type(tweet_text))
cleaned_tweet = clean(tweet_text)
print("Original tweet: ", tweet_text)
print("Cleaned tweet: ", cleaned_tweet)
# create tokens
tweet_tokens = tokenize(cleaned_tweet)

data['cleaned'] = data['text'].apply(clean)
data["tokenized"] = data["cleaned"].apply(tokenize)

# View the result
print(data[['text', 'cleaned', 'tokenized', 'label']].head())
# build bocabulary

vocabulary = set()

tokenized_tweets = data["tokenized"]

for tweet in tokenized_tweets:
    for word in tweet:
        vocabulary.add(word)

print(str(len(vocabulary)))

sorted_vocabulary = sorted(vocabulary)

word_to_ix = {}
ix_to_word = {}

for i in range(len(sorted_vocabulary)):
  word_to_ix.setdefault(sorted_vocabulary[i], i)
  ix_to_word.setdefault(i, sorted_vocabulary[i])

# training preparation
# Utilizziamo il tuo word_to_ix già creato nel file model.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data_for_torch(df, word_map):
    all_indices = []
    offsets = [0]
    labels = []
    
    for _, row in df.iterrows():
        # Convertiamo i token in indici usando il dizionario creato prima
        indices = [word_map[w] for w in row['tokenized'] if w in word_map]
        if len(indices) == 0: continue
        
        all_indices.append(torch.tensor(indices, dtype=torch.int64))
        # Sentiment140: assicurati che le label siano 0 (neg) e 1 (pos)
        l = 1 if row['label'] == 4 else row['label']
        labels.append(l)
        offsets.append(len(indices))
        
    text_tensor = torch.cat(all_indices)
    offsets_tensor = torch.tensor(offsets[:-1]).cumsum(dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    
    return text_tensor, offsets_tensor, label_tensor

# Preparazione
X_text, X_offsets, Y_labels = prepare_data_for_torch(data, word_to_ix)
# Iperparametri
VOCAB_SIZE = len(word_to_ix)
EMBED_DIM = 64
NUM_CLASS = 2 # 0: Negative, 1: Positive

model = SentimentPyTorch(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Fase di Training
model.train()
for epoch in range(35):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X_text.to(device), X_offsets.to(device))
    loss = criterion(output, Y_labels.to(device))
    
    # Backward pass e ottimizzazione
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Modello PyTorch addestrato localmente.")

# TEST TEST

# 1. Load and flatten the test data
with open(test_path, 'r') as f:
    raw_test_data = json.load(f)

flattened_test_data = []
for user in raw_test_data['users']:
    user_tweets = raw_test_data['user_data'][user]['x']
    user_labels = raw_test_data['user_data'][user]['y']
    
    for tweet, label in zip(user_tweets, user_labels):
        flattened_test_data.append({
            'text': tweet[4],
            'label': label
        })

test_data = pd.DataFrame(flattened_test_data)

# 2. Pre-process the test data (Clean and Tokenize)
test_data['cleaned'] = test_data['text'].apply(clean)
test_data['tokenized'] = test_data['cleaned'].apply(tokenize)

# 3. Prepare Tensors for the test set
# We use the existing word_to_ix from the training phase
X_test_text, X_test_offsets, Y_test_labels = prepare_data_for_torch(test_data, word_to_ix)

# 4. Evaluate the model
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for efficiency
    # Move tensors to the same device as the model
    test_output = model(X_test_text.to(device), X_test_offsets.to(device))
    
    # Get predictions (index of the max logit)
    predictions = test_output.argmax(1)
    
    # Calculate accuracy
    correct = (predictions == Y_test_labels.to(device)).float().sum()
    accuracy = correct / len(Y_test_labels)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Test Samples: {len(test_data)}")
    print(f"Test Accuracy: {accuracy.item():.4f}")