import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn.functional as F
import pandas as pd
import os
import csv

if torch.cuda.is_available():
    print("CUDA disponible ✅")
    try:
        torch.nn.functional.scaled_dot_product_attention(
            torch.randn(1, 12, 128, 64).cuda(),
            torch.randn(1, 12, 128, 64).cuda(),
            torch.randn(1, 12, 128, 64).cuda()
        )
        print("Flash Attention activé ✅")
    except Exception as e:
        print("Flash Attention indisponible :", e)
else:
    print("CUDA non disponible ❌")

# Paramètres globaux
MODEL_PATH = "./bert_multilingual_classification"
DATASET_PATH = "./dataset.csv"
LEARNING_RATE = 1e-5
MAX_LENGTH = 128

# Charger le modèle et le tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Charger ou créer le dataset
if not os.path.exists(DATASET_PATH):
    print("Le fichier dataset.csv est manquant. Création d'un nouveau fichier...")
    with open(DATASET_PATH, "w") as f:
        f.write('"text","label"\n')

# Fonction pour prédire une phrase avec orientation
def predict_with_orientation(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    # Prédiction
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence_score = probabilities[0][predicted_class].item()

    # Calcul du score d'orientation
    orientation_map = {0: 0, 1: 1, 2: -1}  # Scores par classe
    orientation_score = orientation_map[predicted_class] * confidence_score

    # Mapper les labels
    label_map = {0: "⚪ Neutre", 1: "🔵 Droite", 2: "🔴 Gauche"}
    return label_map[predicted_class], confidence_score, orientation_score, probabilities.cpu().numpy()

# Fonction pour fine-tuner le modèle avec une nouvelle phrase
def fine_tune_model(text, label):
    model.train()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = torch.tensor([label]).to(device)

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✅ Fine-tuning effectué avec succès. Perte : {loss.item():.4f}")
    
# Ajouter la correction au dataset
def add_to_dataset(text, label):
    existing_data = set()
    with open(DATASET_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_ALL)
        for row in reader:
            if len(row) >= 2:
                existing_data.add((row[0], row[1]))

    if (text, str(label)) in existing_data:
        print(f"⚠️ La phrase existe déjà dans le dataset : \"{text}\",\"{label}\"")
        return

    with open(DATASET_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow([text, str(label)])

    print(f"✅ Phrase ajoutée au dataset : \"{text}\",\"{label}\"")

# Menu interactif
def interactive_menu():
    print("🎉 Bienvenue dans le menu interactif de test et correction BERT 🎉\n")
    while True:
        text = input("Entrez une phrase (ou 'exit' pour quitter) : ").strip()
        if text.lower() == 'exit':
            print("👋 Au revoir !")
            break

        predicted_label, confidence, orientation, probabilities = predict_with_orientation(text)
        print(f"🔍 Prédiction : {predicted_label} (Score de confiance : {confidence:.2f})")
        print(f"🌡️ Degré d'orientation : {orientation:.2f}")

        is_correct = input("La prédiction est-elle correcte ? (y/n) : ").strip().lower()
        if is_correct == "y":
            print("👍 Super, prédiction correcte !")
            add_to_dataset(text, probabilities.argmax().item())

        elif is_correct == "n":
            print("❌ Prédiction incorrecte. Veuillez entrer le bon label :")
            print("0 = Neutre, 1 = Droite, 2 = Gauche")
            correct_label = int(input("Entrez le label correct : ").strip())
            if correct_label not in [0, 1, 2]:
                print("⚠️ Label invalide. Réessayez.\n")
                continue

            fine_tune_model(text, correct_label)
            add_to_dataset(text, correct_label)
        else:
            print("⚠️ Entrée invalide. Répondez par 'y' ou 'n'.\n")

# Lancer le menu interactif
if __name__ == "__main__":
    interactive_menu()