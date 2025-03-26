import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Chemin vers le modèle sauvegardé après entraînement
MODEL_PATH = "./bert_multilingual_classification"

# Charger le tokenizer et le modèle
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Mettre le modèle sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Fonction pour prédire une phrase
def predict_sentence(text):
    # Préparer la phrase pour le modèle
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Envoyer les inputs sur le bon device

    # Obtenir la prédiction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Mapper les labels vers des catégories
    label_map = {0: "Neutre", 1: "Droite", 2: "Gauche"}
    return label_map[predicted_class]

def predict_with_orientation(text):
    # Préparer les données
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Obtenir les logits et les probabilités
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
    label_map = {0: "Neutre", 1: "Droite", 2: "Gauche"}
    return label_map[predicted_class], confidence_score, orientation_score

text = "Les immigrés font la richesse de notre pays"
predicted_label, confidence, orientation = predict_with_orientation(text)
print(f"Texte : {text}")
print(f"Prédiction : {predicted_label} (Score de confiance : {confidence:.2f})")
print(f"Degré d'orientation : {orientation:.2f}")

