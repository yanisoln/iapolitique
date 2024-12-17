import csv
import sys
import re

def structurer_csv(fichier_csv):
    try:
        # Lecture du fichier CSV
        with open(fichier_csv, mode='r', newline='', encoding='utf-8') as fichier:
            lignes = fichier.readlines()

        # Structuration du fichier CSV en respectant les guillemets
        nouvelles_lignes = []
        for ligne in lignes:
            # Utilisation de regex pour détecter les espaces en dehors des guillemets
            nouvelle_ligne = re.sub(r'\s(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', '\n', ligne.strip())
            nouvelles_lignes.append(nouvelle_ligne)

        # Écriture du fichier restructuré
        with open(fichier_csv, mode='w', newline='', encoding='utf-8') as fichier:
            fichier.write('\n'.join(nouvelles_lignes))
            
        print(f"Le fichier {fichier_csv} a été restructuré avec succès.")

    except FileNotFoundError:
        print(f"Erreur : le fichier '{fichier_csv}' est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

def afficher_csv(fichier_csv):
    try:
        # Lecture et affichage du fichier CSV
        with open(fichier_csv, mode='r', newline='', encoding='utf-8') as fichier:
            lecteur_csv = csv.reader(fichier)
            for ligne in lecteur_csv:
                print(ligne)
    except FileNotFoundError:
        print(f"Erreur : le fichier '{fichier_csv}' est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python script.py <emplacement_fichier_csv>")
    else:
        fichier_csv = sys.argv[1]
        structurer_csv(fichier_csv)
        print("\nContenu du fichier CSV :")
        afficher_csv(fichier_csv)
