import re
import pandas as pd
import csv

# Fonction pour ajouter une négation
def negate_sentence(sentence):
    # Ajouter "ne ... pas" autour d'un verbe conjugué ou d'un infinitif (simplifié)
    negated = re.sub(
        r'\b(va|doit|peut|est|sera|ont|a|sont|avait|faut|faut-il|fait|pouvait|veut|voudrait)\b',
        r'ne \1 pas',
        sentence,
        flags=re.IGNORECASE
    )
    if negated == sentence:  # Si aucun verbe n'est détecté
        negated = "Il n'est pas vrai que " + sentence.lower()  # Fallback
    return negated

# Charger le dataset
df = pd.read_csv("./dataset.csv", header=None, names=["text", "label"])

# Filtrer les lignes où le label est "0"
df = df[df['label'] != "0"]

# Générer des phrases négatives et doubler le dataset
df_negated = df.copy()
df_negated['text'] = df_negated['text'].apply(negate_sentence)

# Fusionner l'original et les phrases négatives
df_augmented = pd.concat([df, df_negated], ignore_index=True)

# Sauvegarder le dataset augmenté avec toutes les valeurs entre guillemets
df_augmented.to_csv(
    "./dataset_augmented.csv",
    index=False,
    header=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"'
)

print("Data augmentation terminée. Taille du dataset :", len(df_augmented))
