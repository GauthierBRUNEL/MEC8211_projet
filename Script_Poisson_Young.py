import os
import numpy as np
import pandas as pd
import re
from scipy.stats import linregress

# Chemin du dossier contenant les fichiers de données
directory = r'C:\Users\fgley\OneDrive\Bureau\SCOLAIRE\PREPA\A- GMC04\TX\3_Experimentation\0_Traction Nous\EXP 2'

# Liste pour stocker tous les DataFrames
dfs = []

# Parcours des fichiers du dossier et sous-dossiers
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.txt') and re.search(r'(LA|LG|NO)', filename):
            file_path = os.path.join(root, filename)

            # Lire le fichier et lui attribuer des noms de colonnes
            df = pd.read_csv(file_path, sep='\t', header=None, names=['Strain [%]', 'Constraint [MPa]'])
            
            # Vérifier si le fichier contient bien 2 colonnes
            if df.shape[1] < 2:
                print(f"⚠️ Fichier {filename} : Colonnes manquantes. Vérifie le format.")
                continue

            # Ajouter le nom de l'éprouvette
            eprouvette_name = os.path.splitext(filename)[0]
            df['Nom Eprouvette []'] = eprouvette_name

            # Calcul des colonnes supplémentaires
            df['Force [N]'] = df['Constraint [MPa]'] * (10) * (4)  # Surface en mm² (ex: 10 mm * 4 mm)
            df['Displacement [mm]'] = df['Strain [%]'] / 100 * 50   # L0 = 50 mm

            # Stocker le DataFrame dans la liste
            dfs.append(df)

# Concaténer tous les DataFrames en un seul
df_final = pd.concat(dfs, ignore_index=True)

# **Calcul du module de Young et du coefficient de Poisson**
results = {}

# Boucle sur chaque éprouvette unique
for eprouvette in df_final['Nom Eprouvette []'].unique():
    df_subset = df_final[df_final['Nom Eprouvette []'] == eprouvette]

    # Sélectionner les données jusqu'à 3% de déformation
    mask = df_subset['Strain [%]'] <= 3.0
    strain_limited = df_subset['Strain [%]'][mask] / 100  # Conversion en fraction
    stress_limited = df_subset['Constraint [MPa]'][mask]

    if len(strain_limited) < 2:
        print(f"⚠️ {eprouvette} : Pas assez de points pour le calcul de E.")
        continue

    # Régression linéaire pour E
    slope, _, _, _, _ = linregress(strain_limited, stress_limited)
    E = slope  # Module de Young (MPa)

    # Approximation du coefficient de Poisson ν (si on a ε_trans ailleurs, on pourra le modifier)
    nu = -0.3  # ⚠️ À modifier si on a la vraie valeur de ε_trans

    # Stocker les résultats
    results[eprouvette] = {"E (MPa)": E, "ν (Poisson)": nu}

    print(f"📊 {eprouvette} → E = {E:.2f} MPa, ν = {nu:.3f}")

# Convertir en DataFrame et afficher
df_results = pd.DataFrame.from_dict(results, orient='index')
print("\n📌 Résumé des résultats :")
print(df_results)

# Data for Poisson's ratio
poisson_data = {
    'LA': [0.620589892, 0.326298969, 0.352281062, 0.470229534, 0.282401443],
    'NO': [0.455697631, 0.236252797, 0.215237678, 0.243939537],
    'LG': [0.557932735, 0.249314372, 0.418024998, 0.265210751, 0.536334161, 0.260806492]
}

import matplotlib.pyplot as plt
import seaborn as sns

# Convertir les résultats E en DataFrame et ajouter une colonne 'Direction'
df_results = pd.DataFrame.from_dict(results, orient='index').reset_index()
df_results.columns = ['Nom Eprouvette', 'E (MPa)', 'ν (Poisson)']
df_results['Direction'] = df_results['Nom Eprouvette'].str.extract(r'(LA|LG|NO)')

# Création des boxplots pour E
plt.figure(figsize=(10, 6))
sns.boxplot(x='Direction', y='E (MPa)', data=df_results, palette='Set2')
plt.title("Distribution du module de Young (E) par direction d'impression")
plt.xlabel("Direction d'impression")
plt.ylabel("E (MPa)")
plt.grid(True)
plt.show()

# Transformation des données du coefficient de Poisson en DataFrame
poisson_list = []
for direction, values in poisson_data.items():
    for value in values:
        poisson_list.append({'Direction': direction, 'ν (Poisson)': value})
df_poisson = pd.DataFrame(poisson_list)

# Création des boxplots pour ν
plt.figure(figsize=(10, 6))
sns.boxplot(x='Direction', y='ν (Poisson)', data=df_poisson, palette='Set1')
plt.title("Distribution du coefficient de Poisson par direction d'impression")
plt.xlabel("Direction d'impression")
plt.ylabel("ν (Poisson)")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# Création des histogrammes pour E (Module de Young)
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Set1", n_colors=len(df_results['Direction'].unique()))
for idx, direction in enumerate(df_results['Direction'].unique()):
    subset = df_results[df_results['Direction'] == direction]['E (MPa)']
    sns.histplot(subset, kde=True, bins=10, label=direction, alpha=0.6, color=colors[idx])

plt.title("Histogramme du Module de Young par direction")
plt.xlabel("E (MPa)")
plt.ylabel("Fréquence")
plt.legend(title="Direction")
plt.grid(True)
plt.show()

# Test de normalité pour chaque direction (E)
for direction in df_results['Direction'].unique():
    subset = df_results[df_results['Direction'] == direction]['E (MPa)']
    stat, p = shapiro(subset)
    print(f"\n📊 Test de Shapiro-Wilk pour E ({direction}) : Stat={stat:.3f}, p={p:.5f}")
    if p > 0.05:
        print("✅ Données conformes à une distribution normale.")
    else:
        print("⚠️ Données potentiellement non normales.")

# Création des histogrammes pour ν (Coefficient de Poisson)
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Set2", n_colors=len(df_poisson['Direction'].unique()))
for idx, direction in enumerate(df_poisson['Direction'].unique()):
    subset = df_poisson[df_poisson['Direction'] == direction]['ν (Poisson)']
    sns.histplot(subset, kde=True, bins=10, label=direction, alpha=0.6, color=colors[idx])

plt.title("Histogramme du Coefficient de Poisson par direction")
plt.xlabel("ν (Poisson)")
plt.ylabel("Fréquence")
plt.legend(title="Direction")
plt.grid(True)
plt.show()

# Test de normalité pour chaque direction (ν)
for direction in df_poisson['Direction'].unique():
    subset = df_poisson[df_poisson['Direction'] == direction]['ν (Poisson)']
    stat, p = shapiro(subset)
    print(f"\n📊 Test de Shapiro-Wilk pour ν ({direction}) : Stat={stat:.3f}, p={p:.5f}")
    if p > 0.05:
        print("✅ Données conformes à une distribution normale.")
    else:
        print("⚠️ Données potentiellement non normales.")

