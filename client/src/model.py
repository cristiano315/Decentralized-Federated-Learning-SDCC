#File for the model definition, and training, to implement
import json
import boto3
import pandas as pd
import numpy as np
#import sklearn
#from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
from collections import Counter
# reduce words
#from nltk.corpus import stopwords
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
from torch.utils.data import TensorDataset, DataLoader
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
    def prepare_dataset(bucket_name, s3_key, seed=42):
        print("[Data Prep] Loading JSON data...")
        
        # Load JSON data from S3
        s3 = boto3.client('s3')
        bucket = bucket_name
        key = s3_key
        response = s3.get_object(Bucket=bucket, Key=key)

        # Read the JSON content from the S3 response
        raw_data = json.loads(response['Body'].read().decode('utf-8'))

        texts, labels = [], []
        for user in raw_data['users']:
            for tweet, label in zip(raw_data['user_data'][user]['x'], raw_data['user_data'][user]['y']):
                texts.append(tweet[4])
                # Ensure labels are 0 (neg) and 1 (pos)
                labels.append(1 if label == 4 else label)

        # ---------------------------------------------------------
        # RIMUOVI RIMOUVI RIMOUVI
        # RIDUZIONE AL 10% DEL DATASET (Prima della tokenizzazione)
        # ---------------------------------------------------------
        percentage = 0.016
        total_original = len(texts)
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=percentage, random_state=seed
        )
        print(f"[Data Prep] Dataset ridotto al {percentage*100}%: da {total_original} a {len(texts)} campioni totali.")
        # FINE RIMUOVI --------------
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
        Training loop con elaborazione in mini-batch per evitare errori Out Of Memory.
        Returns the trained model and the number of samples trained on.
        """
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

        # Creazione dei DataLoader per processare i dati in piccoli lotti
        batch_size = 32 # Abbassa a 16, anche 8 se dovessi avere ancora problemi di memoria
        
        train_dataset = TensorDataset(X_train, Mask_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, Mask_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Early Stopping Variables
        patience = 4
        max_epochs = 30
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        num_training_samples = len(Y_train)

        print(f"[Train] Starting local training on {num_training_samples} samples with batch size {batch_size}, Totale batch per epoca: {len(train_loader)}...")
        
        # Fase di Training
        for epoch in range(max_epochs):
            model.train()
            total_train_loss = 0.0
            
            # Iterazione sui batch di addestramento
            for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(train_loader):
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x, batch_mask)
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item() * batch_x.size(0)

                # -----------------------------------------------------
                # CONTROLLO A GRANA FINE: Stampa log ogni 10 batch
                # -----------------------------------------------------
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"[Train] Epoch {epoch+1:02d} | Batch {batch_idx+1:04d}/{len(train_loader):04d} | Current Batch Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / num_training_samples

            # Fase di Validazione a lotti
            model.eval()
            total_val_loss = 0.0
            correct_preds = 0
            
            with torch.no_grad():
                for batch_x, batch_mask, batch_y in val_loader:
                    batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                    
                    val_output = model(batch_x, batch_mask)
                    loss = criterion(val_output, batch_y)
                    total_val_loss += loss.item() * batch_x.size(0)
                                    
                    val_preds = val_output.argmax(1)
                    correct_preds += (val_preds == batch_y).float().sum().item()
            
            avg_val_loss = total_val_loss / len(Y_val)
            val_acc = correct_preds / len(Y_val)
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early Stopping Logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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