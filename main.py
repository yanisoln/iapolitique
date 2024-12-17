import torch
print("CUDA disponible :", torch.cuda.is_available())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np

# Paramètres globaux
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
MODEL_NAME = "bert-base-multilingual-cased"

# 1. Chargement du dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# 2. Préparation des données
print("🔄 Chargement du dataset...")
# Charger le dataset en ignorant l'en-tête
df = pd.read_csv("./dataset.csv", header=0, names=["text", "label"])

# Supprimer d'éventuels espaces dans les noms de colonnes
df.columns = df.columns.str.strip()

# Conversion des labels en entiers
df['label'] = df['label'].astype(int)
# Vérifier les données
assert df.isnull().sum().sum() == 0, "Le dataset contient des valeurs manquantes."
assert df['label'].isin([0, 1, 2]).all(), "Les labels doivent être 0, 1 ou 2."

# Split train / test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 3. Modèle et optimisation
print("🔧 Initialisation du modèle...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

# 4. Fonction d'évaluation
def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    return acc, f1

# 5. Boucle d'entraînement
print("🚀 Début de l'entraînement...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Époque {epoch+1}/{EPOCHS}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1 = evaluate(model, val_loader)

    print(f"✅ Époque {epoch+1} terminée | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

# 6. Sauvegarde du modèle
print("💾 Sauvegarde du modèle...")
model.save_pretrained("./bert_multilingual_classification")
tokenizer.save_pretrained("./bert_multilingual_classification")
print("🎉 Entraînement terminé et modèle sauvegardé !")
