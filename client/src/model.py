#File for the model definition, and training, to implement
import json
import pandas as pd
import numpy as np
#import sklearn
#from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
# reduce words
from nltk.corpus import stopwords
#from sklearn.tree import DecisionTreeClassifier
import copy
from sklearn.model_selection import train_test_split
import subprocess
import os

# testing
import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import PretrainedEmbeddings


class SentimentPyTorch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, word_to_id, embedding_file_path=None):
        super(SentimentPyTorch, self).__init__()
        
        # 1. Use the highly optimized EmbeddingBag
        if embedding_file_path is not None:
            pretrained = PretrainedEmbeddings(embedding_file_path, word_to_id, embed_dim)
            # freeze=False means the model can still tweak the GloVe weights during training
            self.embedding = nn.EmbeddingBag.from_pretrained(
                pretrained.embed.weight, 
                mode='mean', 
                freeze=False
            )
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean', sparse=False)
            
        self.fc = nn.Linear(embed_dim, num_class)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    @staticmethod
    def tokenize(tweet_text):
        tknzr = TweetTokenizer()
        tweet_tokens = tknzr.tokenize(tweet_text)
        return tweet_tokens

    @staticmethod
    def tweet_to_tensor(tokenized_tweet, word_to_idx):
        # Converte i token in indici, ignorando parole fuori vocabolario
        indices = [word_to_idx[w] for w in tokenized_tweet if w in word_to_idx]
        return torch.tensor(indices, dtype=torch.int64)

    @staticmethod
    def clean(text):
        punct = {'💁', '[', '’', '~', '💪', '📚', '🏡', '-', '🐣', '🇺', '”', '̶', '\u200a', ';', '🍕', ' ', '!', '%', ',', '👇', '®', '🌈', '?', '🏽', '=', '💨', '✅', '✔', ')', '|', '‘', '\xa0', '🗽', '&', '🏼', '¿', '…', '🎓', '👉', '❌', '🎧', '👈', '🚂', '+', '🤖', '👎', '→', '¡', '🤔', '️', '👸', '@', '🇸', ':', '“', '•', '🏿', '🏻', '👀', '👏', '—', ']', '✓', '"', '\u200b', '🎤', '\n', '.', '(', '$', '❤', '⬇', '#', '👍', "'", '/', '*', '🏾', '–', '👿'}
        punct.remove(' ')  # keep spaces
        punct.remove('#')  # keep hashtags
        punct.remove('@')  # keep mentions
        punct.remove('\'') # keep single quotes (in order to retain I'm, isn't, etc.)
        temp_text = text.lower()
        temp_text = re.sub(r'https?://\S+', '', temp_text)
        temp_text = re.sub(r'\d+', '0', temp_text)
        temp_text = temp_text.replace('’', '\'')  # some single quotes are slanted, and we want to retain them
        for p in punct:
            temp_text = temp_text.replace(p, ' ')
        temp_text = re.sub(r'\s+', ' ', temp_text)
        cleaned_text = temp_text.strip()
        return cleaned_text

    @staticmethod
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

    @staticmethod
    def prepare_dataset(file_path, seed=42):
        """
        Loads the dataset, cleans it, builds the vocabulary, 
        and prepares the PyTorch tensors. Also downloads GloVe if missing.
        Run this ONCE before the federated loop begins.
        """
        print("[Data Prep] Loading and flattening JSON data...")
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        flattened_data = []
        for user in raw_data['users']:
            user_tweets = raw_data['user_data'][user]['x']
            user_labels = raw_data['user_data'][user]['y']
            
            for tweet, label in zip(user_tweets, user_labels):
                flattened_data.append({
                    'text': tweet[4],  
                    'label': label
                })

        data = pd.DataFrame(flattened_data)

        print("[Data Prep] Cleaning and tokenizing tweets...")
        data['cleaned'] = data['text'].apply(SentimentPyTorch.clean)
        data['tokenized'] = data['cleaned'].apply(SentimentPyTorch.tokenize)

        print("[Data Prep] Building vocabulary...")
        vocabulary = set()
        for tweet in data['tokenized']:
            for word in tweet:
                vocabulary.add(word)

        sorted_vocabulary = sorted(list(vocabulary))
        word_to_ix = {word: i for i, word in enumerate(sorted_vocabulary)}
        print(f"[Data Prep] Vocabulary size: {len(word_to_ix)}")

        print("[Data Prep] Splitting and converting to tensors...")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=seed)
        
        X_train, Off_train, Y_train = SentimentPyTorch.prepare_data_for_torch(train_df, word_to_ix)
        X_val, Off_val, Y_val = SentimentPyTorch.prepare_data_for_torch(val_df, word_to_ix)

        # Download GloVe if it doesn't exist
        GLOVE_PATH = 'glove.6B.50d.txt'
        if not os.path.isfile(GLOVE_PATH):
            print("[Data Prep] Downloading GloVe embeddings...")
            commands = [
                "wget \"https://www.dropbox.com/s/lc3yjhmovq7nyp5/glove6b50dtxt.zip?dl=1\" -O glove6b50dtxt.zip",
                "unzip -o glove6b50dtxt.zip",
                "rm glove6b50dtxt.zip"
            ]
            for command in commands:
                subprocess.run(command, shell=True, executable="/bin/bash")

        return X_train, Off_train, Y_train, X_val, Off_val, Y_val, word_to_ix, GLOVE_PATH

    @staticmethod
    def train_local(model, X_train, Off_train, Y_train, X_val, Off_val, Y_val, device):
        """
        Pure training loop. Takes the current global model and local data tensors.
        Returns the trained model and the number of samples trained on.
        """
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Early Stopping Variables
        patience = 4
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        num_training_samples = len(Y_train) # Y_train length represents number of tweets

        print(f"[Train] Starting local training on {num_training_samples} samples...")
        
        # Fase di Training
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X_train.to(device), Off_train.to(device))
            loss = criterion(output, Y_train.to(device))
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            # Validation Phase
            model.eval()
            with torch.no_grad():
                val_output = model(X_val.to(device), Off_val.to(device))
                val_loss = criterion(val_output, Y_val.to(device)).item()
                                
                # Calculate Val Accuracy for monitoring
                val_preds = val_output.argmax(1)
                val_acc = (val_preds == Y_val.to(device)).float().mean().item()
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                print(f"--> No improvement. Patience: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                model.load_state_dict(best_model_state)
                break

        return model, num_training_samples

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