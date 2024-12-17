import pandas as pd
import re

# Charger le fichier CSV
file_path = "dataset.csv"
try:
    data = pd.read_csv(file_path, header=None, names=["text", "label"])
except Exception as e:
    print(f"Erreur de chargement du fichier : {e}")
    exit()

print("Distribution des labels :")
print(data['label'].value_counts())

# Fonction de validation des donnees
def validate_data(df):
    invalid_rows = []
    for index, row in df.iterrows():
        text = row["text"]
        if index == 0:  # Ignorer l'en-tête
            continue
        label = row["label"]
        # Vérifier si le texte est non vide et que le label est valide
        if not isinstance(text, str) or text.strip() == "":
            invalid_rows.append((index, "Texte vide ou manquant"))
        elif label not in ['0', '1', '2']:
            invalid_rows.append((index, f"Label invalide: {label}"))
    return invalid_rows

# Valider les donnees
invalid_lines = validate_data(data)
if len(invalid_lines) == 0:
    print("\nToutes les lignes sont valides.")
else:
    print(f"\n{len(invalid_lines)} ligne(s) invalide(s) détectée(s) :")
    for idx, error in invalid_lines:
        print(f"Ligne {idx + 1}: {error} | Contenu : {data.iloc[idx]}")

# Recherche et affichage des doublons
duplicates = data[data.duplicated(keep=False)]  # Garder tous les doublons
if not duplicates.empty:
    print("\nDoublons détectés :", len(duplicates))
    print(duplicates)
else:
    print("\nAucun doublon détecté.")

# Statistiques supplémentaires
# 1. Recherche des négations "ne ... pas"
def count_negations(text):
    return len(re.findall(r"\bne\b.*?\bpas\b", text, re.IGNORECASE))

data["negations"] = data["text"].apply(lambda x: count_negations(x) if isinstance(x, str) else 0)
print("\nNombre total de négations (ne ... pas) :", data["negations"].sum())

# 2. Longueur moyenne des textes
data["text_length"] = data["text"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
print("\nLongueur moyenne des textes (en mots) :", round(data["text_length"].mean(), 2))

# 3. Fréquence des mots-clés "oui" et "non"
def count_word(word, text):
    return text.lower().split().count(word) if isinstance(text, str) else 0

# 4. Distribution des longueurs des textes
print("\nDistribution des longueurs des textes (en mots) :")
print(data["text_length"].describe())

# Afficher un résumé des négations pour les 5 premiers textes
print("\nExemple de textes avec leurs statistiques :")
print(data[["text", "negations", "text_length"]].head())
