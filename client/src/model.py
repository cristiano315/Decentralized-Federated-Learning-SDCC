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
        # SELEZIONE DEL DATASET CIRCOLARE BASATA SUL CLIENT ID
        # ---------------------------------------------------------
        client_id = int(os.getenv("CLIENT_ID", 1))
        num_training_nodes = int(os.getenv("TRAINING_NODES", 5))
        training_set_percentage = float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7))
        max_samples_per_client = 1000
        total_original = len(texts)
        training_length = int(total_original * training_set_percentage)
        # divisione intera tra numero originale e il numero di dati di training
        real_samples_per_client = (training_length // num_training_nodes)
        truncated_training_length = real_samples_per_client * num_training_nodes

        samples_per_client = real_samples_per_client if real_samples_per_client <= max_samples_per_client else max_samples_per_client

        # Usiamo il modulo % per far ripartire gli indici dall'inizio se superano il totale
        start_idx = ((client_id - 1) * real_samples_per_client) % (truncated_training_length)
        end_idx = start_idx + samples_per_client

        # Taglio circolare nel caso in cui il blocco superi la fine della lista
        if end_idx <= truncated_training_length:
            texts = texts[start_idx:end_idx]
            labels = labels[start_idx:end_idx]
        else:
            # Prende la parte finale e la unisce con la parte iniziale
            remainder = end_idx - truncated_training_length
            texts = texts[start_idx:] + texts[:remainder]
            labels = labels[start_idx:] + labels[:remainder]

        print(f"[Data Prep] Client {client_id}: estratti {len(texts)} campioni (Start index: {start_idx}).")
        # ---------------------------------------------------------
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
        max_epochs = 5 #changed to 5 for testing, can be increased to 10 or more
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

            '''
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
            '''

        return model, num_training_samples

    @staticmethod
    def prepare_eval_dataset(bucket_name, s3_key, training_nodes):
        """Carica e tokenizza i dati per la valutazione, pescando dalla porzione NON usata per il training."""
        print("[Global Eval] Caricamento intero dataset JSON da S3...")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        raw_data = json.loads(response['Body'].read().decode('utf-8'))

        texts, labels = [], []
        for user in raw_data['users']:
            for tweet, label in zip(raw_data['user_data'][user]['x'], raw_data['user_data'][user]['y']):
                texts.append(tweet[4])
                labels.append(1 if label == 4 else label)

        # ---------------------------------------------------------
        # SELEZIONE DEL DATASET DI VALUTAZIONE (Zero Data Leakage)
        # ---------------------------------------------------------
        total_original = len(texts)
        
        # Ricostruiamo il limite del set di training per evitare sovrapposizioni
        training_set_percentage = float(os.getenv("TRAINING_SET_PERCENTAGE", 0.7))
        client_id = int(os.getenv("CLIENT_ID", 1))
        eval_set_percentage = 1 - training_set_percentage
        eval_length = int(total_original * eval_set_percentage)
        eval_samples_per_client = eval_length // training_nodes

        training_length = int(total_original * training_set_percentage)
        real_samples_per_client = (training_length // training_nodes)
        truncated_training_length = real_samples_per_client * training_nodes

        print(f"[Global Eval] Totale campioni: {total_original}. Indice massimo toccato dal training: {truncated_training_length}.")
        
        # Prendiamo i dati partendo ESATTAMENTE dalla fine del blocco di training!
        start_idx = truncated_training_length + (eval_samples_per_client * (client_id - 1))
        end_idx = start_idx + eval_samples_per_client

        # Controlliamo di non sforare la fine del dataset originale
        if end_idx > total_original:
            print(f"[WARNING] Il 10% dei dati supera il limite del dataset! Verranno usati i {total_original - start_idx} campioni finali rimanenti.")
            end_idx = total_original

        eval_texts = texts[start_idx:end_idx]
        eval_labels = labels[start_idx:end_idx]

        print(f"[Global Eval] Selezionati {len(eval_texts)} campioni 'unseen' per la valutazione (dall'indice {start_idx} al {end_idx}).")
        # ---------------------------------------------------------

        print(f"[Global Eval] Tokenizzazione in corso...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encoded = tokenizer(eval_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

        X = encoded['input_ids']
        Mask = encoded['attention_mask']
        Y = torch.tensor(eval_labels, dtype=torch.int64)

        return X, Mask, Y

    @staticmethod
    def evaluate_global(model, X_test, Mask_test, Y_test, device):
        """Calcola Loss e Accuracy su tutto il dataset fornito."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Usiamo un batch_size come in train_local per evitare Out Of Memory
        batch_size = 32 
        test_dataset = TensorDataset(X_test, Mask_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        correct_preds = 0
        
        print(f"[Global Eval] Avvio inferenza su {len(Y_test)} campioni. Batch totali: {len(test_loader)}")
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(test_loader):
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                
                output = model(batch_x, batch_mask)
                loss = criterion(output, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                                
                preds = output.argmax(1)
                correct_preds += (preds == batch_y).float().sum().item()
                
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                    print(f"[Global Eval] Batch {batch_idx+1:04d}/{len(test_loader):04d} completato.")
        
        avg_loss = total_loss / len(Y_test)
        accuracy = correct_preds / len(Y_test)
        
        print("\n" + "="*30)
        print(f"[RISULTATI GLOBALI] Loss Finale: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        print("="*30 + "\n")
        
        return avg_loss, accuracy