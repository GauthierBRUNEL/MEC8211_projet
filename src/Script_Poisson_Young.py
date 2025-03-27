import os
import numpy as np
import pandas as pd
import re
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc

HISTO = False

# Chemin du dossier contenant les fichiers de données
directory = r'C:\Users\fgley\OneDrive\Bureau\SCOLAIRE\PREPA\A- GMC04\TX\3_Experimentation\0_Traction Nous\EXP 2'

# Liste pour stocker tous les DataFrames
dfs = []

#%% Lire et assigner toutes les données 
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

#%% **Calcul du module de Young et du coefficient de Poisson**
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

#%% Plot Figure 1 à 3 

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

for eprouvette in df_final['Nom Eprouvette []'].unique():
    df_subset = df_final[df_final['Nom Eprouvette []'] == eprouvette]
    mask = df_subset['Strain [%]'] <= 3.0
    strain_limited = df_subset['Strain [%]'][mask] / 100
    stress_limited = df_subset['Constraint [MPa]'][mask]

    if len(strain_limited) < 2:
        print(f"⚠️ {eprouvette} : Pas assez de points pour le calcul de E.")
        continue

    slope, _, _, _, _ = linregress(strain_limited, stress_limited)
    E = slope  # Module de Young (MPa)
    nu = -0.3  # Coefficient de Poisson approximé

    # Calcul du coefficient de cisaillement
    G = E / (2 * (1 + nu))

    results[eprouvette] = {"E (MPa)": E, "ν (Poisson)": nu, "G (MPa)": G}

    print(f"📊 {eprouvette} → E = {E:.2f} MPa, ν = {nu:.3f}, G = {G:.2f} MPa")

# Résultats finaux
df_results = pd.DataFrame.from_dict(results, orient='index').reset_index()
df_results.columns = ['Nom Eprouvette', 'E (MPa)', 'ν (Poisson)', 'G (MPa)']
df_results['Direction'] = df_results['Nom Eprouvette'].str.extract(r'(LA|LG|NO)')

# Boxplot pour le coefficient de cisaillement G
plt.figure(figsize=(10, 6))
sns.boxplot(x='Direction', y='G (MPa)', data=df_results, palette='Set3')
plt.title("Distribution du coefficient de cisaillement (G) par direction d'impression")
plt.xlabel("Direction d'impression")
plt.ylabel("G (MPa)")
plt.grid(True)
plt.show()

#%% Test aléatoire ou epistémique

from scipy.stats import shapiro, kstest, norm

# Test de normalité et visualisation par direction
for variable in ['E (MPa)', 'ν (Poisson)', 'G (MPa)']:
    for direction in df_results['Direction'].unique():
        data = df_results[(df_results['Direction'] == direction)][variable].dropna()

        if len(data) < 2 or data.nunique() == 1:
            print(f"⚠️ {variable} en direction {direction} : Données insuffisantes ou trop uniformes pour une analyse correcte.")
            continue

        plt.figure(figsize=(10, 6))
        
        # Histogramme et courbe KDE avec ajustement si nécessaire
        sns.histplot(data, kde=True, bins=15, color='skyblue', kde_kws={'bw_adjust': 0.5})
        plt.title(f'Distribution de {variable} - Direction {direction}')
        plt.xlabel(variable)
        plt.grid(True)
        plt.show()

        # Test de Shapiro-Wilk
        stat, p_value = shapiro(data)
        if p_value > 0.05:
            print(f"✅ {variable} en direction {direction} suit potentiellement une loi normale (p = {p_value:.3f})")
        else:
            print(f"❌ {variable} en direction {direction} ne suit PAS une loi normale (p = {p_value:.3f})")

        # Test de Kolmogorov-Smirnov
        stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        if p_value > 0.05:
            print(f"✅ {variable} en direction {direction} est conforme à une loi normale d'après KS (p = {p_value:.3f})")
        else:
            print(f"❌ {variable} en direction {direction} n'est PAS conforme à une loi normale d'après KS (p = {p_value:.3f})")


# PDF et CDF pour les variables suivant une loi normale
for variable in ['E (MPa)', 'ν (Poisson)', 'G (MPa)']:
    for direction in df_results['Direction'].unique():
        data = df_results[(df_results['Direction'] == direction)][variable].dropna()

        if len(data) < 2 or data.nunique() == 1:
            print(f"⚠️ {variable} en direction {direction} : Données insuffisantes ou trop uniformes pour une analyse correcte.")
            continue

        # Estimation des paramètres de la loi normale
        mu, sigma = data.mean(), data.std()

        # Affichage des paramètres
        print(f"📈 {variable} - Direction {direction} : μ = {mu:.2f}, σ = {sigma:.2f}")

        # Tracé de la PDF
        x = np.linspace(min(data), max(data), 100)
        pdf = norm.pdf(x, mu, sigma)

        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=False, bins=15, color='skyblue', stat='density')
        plt.plot(x, pdf, label='PDF - Loi Normale', color='red')
        plt.title(f'Distribution de {variable} - Direction {direction}')
        plt.xlabel(variable)
        plt.legend()
        plt.grid(True)
        plt.show()

        # Tracé de la CDF
        cdf = norm.cdf(x, mu, sigma)

        plt.figure(figsize=(10, 6))
        plt.plot(x, cdf, label='CDF - Loi Normale', color='green')
        plt.title(f'Fonction de répartition cumulative de {variable} - Direction {direction}')
        plt.xlabel(variable)
        plt.ylabel('Probabilité cumulative')
        plt.legend()
        plt.grid(True)
        plt.show()

#%% Plot des variables epistémique 

# Tracé des CDF en escalier avec légende et couleurs
colors = {'LA': 'blue', 'NO': 'green', 'LG': 'orange'}  # Couleurs par direction

for direction in df_poisson['Direction'].unique():
    plt.figure(figsize=(8, 6))

    data = df_poisson[df_poisson['Direction'] == direction]['ν (Poisson)']

    # Bornes min et max
    lower_bound = np.min(data)
    upper_bound = np.max(data)

    # Tracé de la fonction escalier pour la borne inférieure (palier à 1 ajouté)
    plt.step([0, lower_bound, lower_bound, upper_bound], [0, 0, 1, 1], 
             color=colors[direction], linestyle='--', label='Minimum value of interval')

    # Tracé de la fonction escalier pour la borne supérieure
    plt.step([0, upper_bound, upper_bound, max(data) + 0.1], [0, 0, 1, 1], 
             color=colors[direction], linestyle='-', label='Maximum value of interval')

    plt.title(f"Fonction de répartition cumulative (CDF) - Direction {direction}")
    plt.xlabel('ν (Poisson)')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.grid(True)

    plt.show()

#%% Formation de couple 

# Nombre de couples à générer
n_samples = 50

ORTOTROP = False

if ORTOTROP == True : 
    # Génération des échantillons LHS
    lhs_samples = qmc.LatinHypercube(d=6).random(n=n_samples)

    # Paramètres et intervalles
    E_params = {k: (df_results[df_results['Direction'] == k]['E (MPa)'].mean(),
                     df_results[df_results['Direction'] == k]['E (MPa)'].std())
                for k in ['LA', 'LG', 'NO']}

    nu_intervals = {k: (min(poisson_data[k]), max(poisson_data[k])) for k in ['LA', 'LG', 'NO']}

    # Transformation LHS et calculs
    E = {k: norm.ppf(lhs_samples[:, i], loc=E_params[k][0], scale=E_params[k][1]) for i, k in enumerate(['LA', 'LG', 'NO'])}
    nu = {k: nu_intervals[k][0] + (nu_intervals[k][1] - nu_intervals[k][0]) * lhs_samples[:, i + 3] for i, k in enumerate(['LA', 'LG', 'NO'])}
    G = {k: E[k] / (2 * (1 + nu[k])) for k in ['LA', 'LG', 'NO']}
    
    df_couples = pd.DataFrame({**E, **nu, **G})
    df_couples.to_csv('Couples_LHS_Complets.csv', index=False)
    print(f"✅ {len(df_couples)} couples générés avec succès et sauvegardés dans 'Couples_LHS_Complets.csv'")
    
else : 
    # Génération des échantillons LHS
    lhs_samples = qmc.LatinHypercube(d=2).random(n=n_samples)  # d=2 car on ne garde que LA
    
    # Paramètres et intervalles pour LA uniquement
    E_mean_LA, E_std_LA = df_results[df_results['Direction'] == 'LA']['E (MPa)'].mean(), df_results[df_results['Direction'] == 'LA']['E (MPa)'].std()
    nu_min_LA, nu_max_LA = min(poisson_data['LA']), max(poisson_data['LA'])
    
    # Transformation LHS et calculs pour LA uniquement
    E_LA = norm.ppf(lhs_samples[:, 0], loc=E_mean_LA, scale=E_std_LA)
    nu_LA = nu_min_LA + (nu_max_LA - nu_min_LA) * lhs_samples[:, 1]
    
    # Création du DataFrame et export CSV
    df_LA = pd.DataFrame({'E_LA': E_LA, 'nu_LA': nu_LA})
    df_LA.to_csv('Couples_LHS_LA.csv', index=False)
    
    print(f"✅ {len(df_LA)} couples générés avec succès et sauvegardés dans 'Couples_LHS_LA.csv'")


