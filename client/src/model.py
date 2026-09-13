#File for the model definition, and training, to implement
import json
import boto3
from sklearn.model_selection import train_test_split
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
        """ inizialize model """
        # load pre-trained DistilBERT for embedding
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # don't fine tune the BERT model, its weights stays the same
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # classification head, only part to train
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
        # pass inputs to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # extract the first token of the sequence for classification
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        # apply neural head only to this first token
        return self.fc(cls_token_state)

    @staticmethod
    def prepare_dataset(bucket_name, s3_key, num_training_nodes, training_set_percentage, max_samples_per_client=1000, seed=42):
        """return training set, validation set and test set"""
        print("Loading JSON data...")
        
        # load JSON data from S3
        s3 = boto3.client('s3')
        bucket = bucket_name
        key = s3_key
        response = s3.get_object(Bucket=bucket, Key=key)
        raw_data = json.loads(response['Body'].read().decode('utf-8'))

        # extract all tweets and labels
        texts, labels = [], []
        for user in raw_data['users']:
            for tweet, label in zip(raw_data['user_data'][user]['x'], raw_data['user_data'][user]['y']):
                texts.append(tweet[4])
                # a label of 4 means the sentiment is positive
                labels.append(1 if label == 4 else label)

        # select client data to mimick client dataset
        client_id = int(os.getenv("CLIENT_ID", 1))
        total_original = len(texts)
        # make sure the dataset length is divisible by the number of clients
        total_samples_per_client = (total_original // num_training_nodes)
        truncated_total_length = total_samples_per_client * num_training_nodes

        # check start and end using client id, the dataset length and the number of samples per client
        start_idx = ((client_id - 1) * total_samples_per_client) % truncated_total_length
        end_idx = start_idx + total_samples_per_client

        # get all the client texts and labels
        if end_idx <= truncated_total_length:
            client_texts = texts[start_idx:end_idx]
            client_labels = labels[start_idx:end_idx]
        else:
            remainder = end_idx - truncated_total_length
            client_texts = texts[start_idx:] + texts[:remainder]
            client_labels = labels[start_idx:] + labels[:remainder]

        print(f"Client {client_id}: extracted {len(client_texts)} instances (Start index: {start_idx}).")
        print("Tokenizing with DistilBERT...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # apply tokenization
        encoded = tokenizer(client_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        # the Y is just the correct prediction, 0 or 1, just make the client_labels list a tensor
        X = encoded['input_ids']
        # part of the text is masked because the BERT training goal is to predict such words
        Mask = encoded['attention_mask']
        Y = torch.tensor(client_labels, dtype=torch.int64)

        print("Splitting dataset...")
        test_size = 1.0 - training_set_percentage
        # test set creation
        X_train, X_test, Mask_train, Mask_test, Y_train, Y_test = train_test_split(
            X, Mask, Y, test_size=test_size, random_state=seed
        )

        # validation set creation
        val_size = 0.1 # fixed dimension
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
        """returns the trained model and the number of samples trained on."""
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        batch_size = 32
        # correct training and validation set representation        
        train_dataset = TensorDataset(X_train, Mask_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, Mask_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # save number of training samples to return it
        num_training_samples = len(Y_train)

        print(f"Starting local training on {num_training_samples} samples with batch size {batch_size}, Total batch per epoch: {len(train_loader)}...")
        
        # training phase
        for epoch in range(num_epochs):
            # model in training mode
            model.train()
            total_train_loss = 0.0
            
            for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(train_loader):
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                
                # optimizer inizialization
                optimizer.zero_grad()
                # forward pass
                output = model(batch_x, batch_mask)
                # criterion applies the softmax
                loss = criterion(output, batch_y)
                # backward pass
                loss.backward()
                # dynamic learning rate for each weight
                optimizer.step()
                # loss for all the batch
                total_train_loss += loss.item() * batch_x.size(0)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1:04d}/{len(train_loader):04d} | Current Batch Loss: {loss.item():.4f}")

            # average training loss for tweet prediction
            avg_train_loss = total_train_loss / num_training_samples

            # validation phase
            model.eval()
            total_val_loss = 0.0
            correct_preds = 0
            
            with torch.no_grad():
                for batch_x, batch_mask, batch_y in val_loader:
                    batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                    # forward pass and loss calculation
                    val_output = model(batch_x, batch_mask)
                    loss = criterion(val_output, batch_y)
                    total_val_loss += loss.item() * batch_x.size(0)
                    # how many predictions correct in this round                                    
                    val_preds = val_output.argmax(1)
                    correct_preds += (val_preds == batch_y).float().sum().item()
            
            avg_val_loss = total_val_loss / len(Y_val)
            val_acc = correct_preds / len(Y_val)
            # print average characteristics
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return model, num_training_samples

    @staticmethod
    def evaluate_global(model, X_test, Mask_test, Y_test, device):
        """returns metrics for binary classification"""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        batch_size = 32
        # as in training, load the dataset
        test_dataset = TensorDataset(X_test, Mask_test, Y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # inizialization for binary task
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

                # probability the tweet is 1
                probs = F.softmax(output, dim=-1)[:, 1]
                # label prediction
                preds = output.argmax(dim=1)

                # metrics update
                # AUC requires the probability
                # the other metrics the actual prediction
                acc_metric.update(preds, batch_y)
                precision_metric.update(preds, batch_y)
                recall_metric.update(preds, batch_y)
                kappa_metric.update(preds, batch_y)
                auc_metric.update(probs, batch_y)
                conf_mat_metric.update(preds, batch_y)

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                    print(
                        f"Batch {batch_idx+1:04d}/{len(test_loader):04d} completed."
                    )
        # extract metrics from all the variables
        metrics = {
            "loss": total_loss / len(Y_test),
            "accuracy": acc_metric.compute().item(),
            "precision": precision_metric.compute().item(),
            "recall": recall_metric.compute().item(),
            "kappa": kappa_metric.compute().item(),
            "auc": auc_metric.compute().item(),
            "confusion_matrix": conf_mat_metric.compute().cpu(),
        }

        # final print of the metrics
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