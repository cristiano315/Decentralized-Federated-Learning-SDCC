#File for the model definition, and training, to implement
import json
import boto3
import pandas as pd
import numpy as np
from collections import Counter
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
import torchmetrics.classification as tm_cls


class SentimentPyTorch(nn.Module):
    def __init__(self, num_class=2):
        super(SentimentPyTorch, self).__init__()
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze the BERT layers so FedAvg only trains the classifier
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification head (DistilBERT hidden size is 768)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, num_class)
        )
        
    def forward(self, input_ids, attention_mask):
        # Pass inputs to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the [CLS] token (first tokeno f the sequence) for classification
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token_state)

    @staticmethod
    def prepare_dataset(bucket_name, s3_key, num_training_nodes, training_set_percentage, max_samples_per_client=1000, seed=42):
        print("Loading JSON data...")
        
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

        # selezione dataset client basata su client id
        client_id = int(os.getenv("CLIENT_ID", 1))
        total_original = len(texts)
        # divisione intera tra numero originale e il numero di dati totali
        # ogni client ha suo blocco dati da cui estrarre training set e test set 
        total_samples_per_client = (total_original // num_training_nodes)
        truncated_total_length = total_samples_per_client * num_training_nodes

        # Usiamo il modulo % per far ripartire gli indici dall'inizio se superano il totale
        start_idx = ((client_id - 1) * total_samples_per_client) % truncated_total_length
        end_idx = start_idx + total_samples_per_client

        # Taglio circolare nel caso in cui il blocco superi la fine della lista
        if end_idx <= truncated_total_length:
            client_texts = texts[start_idx:end_idx]
            client_labels = labels[start_idx:end_idx]
        else:
            # Prende la parte finale e la unisce con la parte iniziale
            remainder = end_idx - truncated_total_length
            client_texts = texts[start_idx:] + texts[:remainder]
            client_labels = labels[start_idx:] + labels[:remainder]

        print(f"Client {client_id}: extracted {len(client_texts)} instances (Start index: {start_idx}).")
        print("Tokenizing with DistilBERT...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # This handles cleaning, tokenizing, and padding all at once
        encoded = tokenizer(client_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

        X = encoded['input_ids']
        Mask = encoded['attention_mask']
        Y = torch.tensor(client_labels, dtype=torch.int64)

        print("Splitting dataset...")
        # porzione dedicata al test set
        test_size = 1.0 - training_set_percentage
        X_train, X_test, Mask_train, Mask_test, Y_train, Y_test = train_test_split(
            X, Mask, Y, test_size=test_size, random_state=seed
        )

        # porzione per validation set
        val_size = 0.1 # quantita fissa
        val_split_idx = int(len(X_train) * (1 - val_size))

        X_val = X_train[val_split_idx:]
        Mask_val = Mask_train[val_split_idx:]
        Y_val = Y_train[val_split_idx:]

        X_train = X_train[:val_split_idx]
        Mask_train = Mask_train[:val_split_idx]
        Y_train = Y_train[:val_split_idx]

        if len(Y_train) > max_samples_per_client:
            X_train = X_train[:max_samples_per_client]
            Mask_train = Mask_train[:max_samples_per_client]
            Y_train = Y_train[:max_samples_per_client]
        
        if len(Y_val) > max_samples_per_client:
            X_val = X_val[:max_samples_per_client]
            Mask_val = Mask_val[:max_samples_per_client]
            Y_val = Y_val[:max_samples_per_client]

        if len(Y_test) > max_samples_per_client:
            X_test = X_test[:max_samples_per_client]
            Mask_test = Mask_test[:max_samples_per_client]
            Y_test = Y_test[:max_samples_per_client]

        return X_train, Mask_train, Y_train, X_val, Mask_val, Y_val, X_test, Mask_test, Y_test

    @staticmethod
    def train_local(model, X_train, Mask_train, Y_train, X_val, Mask_val, Y_val, device, num_epochs):
        """
        Training loop con elaborazione in mini-batch per evitare errori Out Of Memory.
        Returns the trained model and the number of samples trained on.
        """
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        batch_size = 32
        
        train_dataset = TensorDataset(X_train, Mask_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, Mask_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        num_training_samples = len(Y_train)

        print(f"Starting local training on {num_training_samples} samples with batch size {batch_size}, Total batch per epoch: {len(train_loader)}...")
        
        # fase di training
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            
            for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(train_loader):
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x, batch_mask)
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item() * batch_x.size(0)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1:04d}/{len(train_loader):04d} | Current Batch Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / num_training_samples

            # fase di validazione
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

        return model, num_training_samples

    @staticmethod
    def evaluate_global(model, X_test, Mask_test, Y_test, device):
        """Calcola Loss, Accuracy, Precision, Recall, Kappa, AUC e Confusion Matrix per classificazione binaria."""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        batch_size = 32
        test_dataset = TensorDataset(X_test, Mask_test, Y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # Inizializzazione fissa per task binario
        acc_metric = tm_cls.BinaryAccuracy().to(device)
        precision_metric = tm_cls.BinaryPrecision().to(device)
        recall_metric = tm_cls.BinaryRecall().to(device)
        kappa_metric = tm_cls.BinaryCohenKappa().to(device)
        auc_metric = tm_cls.BinaryAUROC().to(device)
        conf_mat_metric = tm_cls.BinaryConfusionMatrix().to(device)

        total_loss = 0.0

        print(
            f"Starting inference on {len(Y_test)} samples (Binary) for global evaluation. Total batches: {len(test_loader)}"
        )

        with torch.no_grad():
            for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(
                test_loader
            ):
                batch_x, batch_mask, batch_y = (
                    batch_x.to(device),
                    batch_mask.to(device),
                    batch_y.to(device),
                )

                output = model(batch_x, batch_mask)
                loss = criterion(output, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                # Probabilità della classe positiva (indice 1) e classe predetta
                probs = F.softmax(output, dim=-1)[:, 1]
                preds = output.argmax(dim=1)

                # Aggiornamento incrementale
                acc_metric.update(preds, batch_y)
                precision_metric.update(preds, batch_y)
                recall_metric.update(preds, batch_y)
                kappa_metric.update(preds, batch_y)
                auc_metric.update(probs, batch_y)
                conf_mat_metric.update(preds, batch_y)

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(
                    test_loader
                ):
                    print(
                        f"Batch {batch_idx+1:04d}/{len(test_loader):04d} completed."
                    )

        metrics = {
            "loss": total_loss / len(Y_test),
            "accuracy": acc_metric.compute().item(),
            "precision": precision_metric.compute().item(),
            "recall": recall_metric.compute().item(),
            "kappa": kappa_metric.compute().item(),
            "auc": auc_metric.compute().item(),
            "confusion_matrix": conf_mat_metric.compute().cpu(),
        }

        print("\n" + "=" * 45)
        print(f"[GLOBAL RESULTS]")
        print(f"Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}")
        print(
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )
        print(f"Cohen's Kappa: {metrics['kappa']:.4f} | ROC-AUC: {metrics['auc']:.4f}")
        print("-" * 45)
        cm = metrics["confusion_matrix"].numpy()
        tn, fp, fn, tp = cm.ravel()

        print("-" * 45)
        print("Confusion Matrix:")
        print(f"               Pred Neg (0)    Pred Pos (1)")
        print(f"Actual Neg (0)     {tn:<12d}    {fp:<12d}  (TN, FP)")
        print(f"Actual Pos (1)     {fn:<12d}    {tp:<12d}  (FN, TP)")
        print("=" * 45 + "\n")
        print("=" * 45 + "\n")

        return metrics