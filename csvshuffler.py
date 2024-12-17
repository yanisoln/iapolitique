import csv
import sys
import random

def melanger_csv(fichier_csv):
    try:
        # Lecture du fichier CSV
        with open(fichier_csv, mode='r', newline='', encoding='utf-8') as fichier:
            lecteur_csv = list(csv.reader(fichier))
            
            if not lecteur_csv:
                print("Le fichier CSV est vide.")
                return
            
            # Enlever les guillemets de l'en-tête
            entete = [col.replace('"', '') for col in lecteur_csv[0]]
            entete = lecteur_csv[0]  # Première ligne (en-tête)
            donnees = lecteur_csv[1:]  # Les autres lignes
            
            # Mélanger les lignes de données
            random.shuffle(donnees)

        # Écriture du nouveau fichier CSV mélangé
        with open(fichier_csv, mode='w', newline='', encoding='utf-8') as fichier:
            ecrivain = csv.writer(fichier, quoting=csv.QUOTE_ALL)
            ecrivain.writerow(entete)  # Réécrire l'en-tête
            ecrivain.writerows(donnees)  # Réécrire les lignes mélangées
            
        print(f"Le fichier {fichier_csv} a été mélangé avec succès.")
    
    except FileNotFoundError:
        print(f"Erreur : le fichier '{fichier_csv}' est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python script.py <emplacement_fichier_csv>")
    else:
        fichier_csv = sys.argv[1]
        melanger_csv(fichier_csv)