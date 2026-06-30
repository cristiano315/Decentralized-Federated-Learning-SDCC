#File for the model definition, and training, to implement
import json
import pandas as pd
import numpy as np
#import sklearn
#from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# reduce words
from nltk.corpus import stopwords
#from sklearn.tree import DecisionTreeClassifier
import copy
from sklearn.model_selection import train_test_split
import subprocess
import os
from transformers import DistilBertModel, DistilBertTokenizer

# testing
import torch
import torch.nn as nn
import torch.nn.functional as F
# from embeddings import PretrainedEmbeddings


class SentimentPyTorch(nn.Module):
    def __init__(self, num_class=2):
        super(SentimentPyTorch, self).__init__()
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze the BERT layers so FedAvg only trains the classifier
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification head (DistilBERT hidden size is 768)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_class)
        
    def forward(self, input_ids, attention_mask):
        # Pass inputs to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the [CLS] token (first tokeno f the sequence) for classification
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token_state)

    @staticmethod
    def prepare_dataset(file_path, seed=42):
        print("[Data Prep] Loading JSON data...")
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        texts, labels = [], []
        for user in raw_data['users']:
            for tweet, label in zip(raw_data['user_data'][user]['x'], raw_data['user_data'][user]['y']):
                texts.append(tweet[4])
                # Ensure labels are 0 (neg) and 1 (pos)
                labels.append(1 if label == 4 else label)

        print("[Data Prep] Tokenizing with DistilBERT...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # This handles cleaning, tokenizing, and padding all at once
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

        X = encoded['input_ids']
        Mask = encoded['attention_mask']
        Y = torch.tensor(labels, dtype=torch.int64)

        print("[Data Prep] Splitting dataset...")
        # PyTorch random split or sklearn train_test_split on the tensors
        X_train, X_val, Mask_train, Mask_val, Y_train, Y_val = train_test_split(
            X, Mask, Y, test_size=0.2, random_state=seed
        )

        return X_train, Mask_train, Y_train, X_val, Mask_val, Y_val

    @staticmethod
    def train_local(model, X_train, Mask_train, Y_train, X_val, Mask_val, Y_val, device):
        """
        Pure training loop. Takes the current global model and local data tensors.
        Returns the trained model and the number of samples trained on.
        """
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

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
            output = model(X_train.to(device), Mask_train.to(device))
            loss = criterion(output, Y_train.to(device))
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            # Validation Phase
            model.eval()
            with torch.no_grad():
                val_output = model(X_val.to(device), Mask_val.to(device))
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