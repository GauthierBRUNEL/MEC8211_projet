# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:52:01 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt

# Nouvelles données organisées en colonnes
data = np.array([
    [0.30586094, 18.68660252, 18686602.52, 0.048504, 0.41870523, 18.68660252, 18686602.52, 0.048364, 0.45907335, 18.68660252, 18686602.52, 0.04823, 0.39583582, 18.68660252, 18686602.52, 0.048415, 0.34591471, 18.68660252, 18686602.52, 0.048484],
    [0.30586094, 17.31103183, 17311031.83, 0.052358, 0.41870523, 17.31103183, 17311031.83, 0.052207, 0.45907335, 17.31103183, 17311031.83, 0.052062, 0.39583582, 17.31103183, 17311031.83, 0.052262, 0.34591471, 17.31103183, 17311031.83, 0.052336],
    [0.30586094, 13.85463287, 13854632.87, 0.06542,  0.41870523, 13.85463287, 13854632.87, 0.065231, 0.45907335, 13.85463287, 13854632.87, 0.065051, 0.39583582, 13.85463287, 13854632.87, 0.0653,   0.34591471, 13.85463287, 13854632.87, 0.065393],
    [0.30586094, 21.93956988, 21939569.88, 0.041312, 0.41870523, 21.93956988, 21939569.88, 0.041193, 0.45907335, 21.93956988, 21939569.88, 0.041079, 0.39583582, 21.93956988, 21939569.88, 0.041236, 0.34591471, 21.93956988, 21939569.88, 0.041295],
    [0.30586094, 15.38049534, 15380495.34, 0.05893,  0.41870523, 15.38049534, 15380495.34, 0.05876,  0.45907335, 15.38049534, 15380495.34, 0.058597, 0.39583582, 15.38049534, 15380495.34, 0.058822, 0.34591471, 15.38049534, 15380495.34, 0.058905],
    [0.30586094, 23.69406535, 23694065.35, 0.038253, 0.41870523, 23.69406535, 23694065.35, 0.038143, 0.45907335, 23.69406535, 23694065.35, 0.038037, 0.39583582, 23.69406535, 23694065.35, 0.038183, 0.34591471, 23.69406535, 23694065.35, 0.038237],
    [0.30586094, 20.71787884, 20717878.84, 0.043748, 0.41870523, 20.71787884, 20717878.84, 0.043622, 0.45907335, 20.71787884, 20717878.84, 0.043501, 0.39583582, 20.71787884, 20717878.84, 0.043668, 0.34591471, 20.71787884, 20717878.84, 0.04373],
    [0.30586094, 19.49344516, 19493445.16, 0.046496, 0.41870523, 19.49344516, 19493445.16, 0.046362, 0.45907335, 19.49344516, 19493445.16, 0.046234, 0.39583582, 19.49344516, 19493445.16, 0.046411, 0.34591471, 19.49344516, 19493445.16, 0.046477],
    [0.30586094, 25.5036786,  25503678.6,  0.035539, 0.41870523, 25.5036786,  25503678.6,  0.035436, 0.45907335, 25.5036786,  25503678.6,  0.035338, 0.39583582, 25.5036786,  25503678.6,  0.035474, 0.34591471, 25.5036786,  25503678.6,  0.035524],
    [0.30586094, 12.70714391, 12707143.91, 0.071328, 0.41870523, 12.70714391, 12707143.91, 0.071122, 0.45907335, 12.70714391, 12707143.91, 0.070925, 0.39583582, 12.70714391, 12707143.91, 0.071197, 0.34591471, 12.70714391, 12707143.91, 0.071298]
])

# Extraction des SRQ pour chaque valeur de nu (1, 2, ..., 5)
SRQ_data = np.zeros((10, 5))  # 10 lignes, 5 colonnes (1 par valeur de nu)

for i in range(5):
    SRQ_data[:, i] = data[:, i * 4 + 3]  # La 4ème colonne de chaque bloc (index 3, 7, 11, etc.)

nu_values = [0.30586094, 0.41870523, 0.45907335, 0.39583582, 0.34591471]

# Tracé des CDFs
plt.figure(figsize=(8, 6))

for i in range(5):
    sorted_data = np.sort(SRQ_data[:, i])
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, linestyle='-', label=f"ν = {nu_values[i]:.8f}")
    plt.scatter(sorted_data, cdf, s=20)

plt.xlabel("SRQ")
plt.ylabel("CDF")
plt.title("Fonctions de Répartition Cumulatives (CDF) des SRQ")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.show()

# Trouver les CDF les plus à gauche et à droite
cdf_left_index = None
cdf_right_index = None

min_SRQ = np.inf  # Pour trouver la CDF la plus à gauche
max_SRQ = -np.inf  # Pour trouver la CDF la plus à droite

for i in range(5):
    data = np.sort(SRQ_data[:, i])
    if data[0] < min_SRQ:
        min_SRQ = data[0]
        cdf_left_index = i
    if data[-1] > max_SRQ:
        max_SRQ = data[-1]
        cdf_right_index = i

# Tracé de la p-box avec les CDF extrêmes
plt.figure(figsize=(8, 6))

for idx, label in zip([cdf_left_index, cdf_right_index], ["Gauche", "Droite"]):
    data = np.sort(SRQ_data[:, idx])
    cdf = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, linestyle='-', label=f"CDF {label} (ν = {nu_values[idx]:.8f})")
    plt.scatter(data, cdf, s=20)

plt.xlabel("SRQ")
plt.ylabel("CDF")
plt.title("P-box : Enveloppe des Fonctions de Répartition Cumulatives")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Extraction des valeurs de E et de ν
E_values = data[:, 1]  # Deuxième colonne du premier bloc
nu_values = np.array([0.30586094, 0.41870523, 0.45907335, 0.39583582, 0.34591471])

# Calcul de la médiane des SRQ
SRQ_median_E = np.median(SRQ_data, axis=1)  # Médiane pour chaque E
SRQ_median_nu = np.median(SRQ_data, axis=0)  # Médiane pour chaque ν

# Régression linéaire pour E
E_values_reshaped = E_values.reshape(-1, 1)
model_E = LinearRegression().fit(E_values_reshaped, SRQ_median_E)
SRQ_pred_E = model_E.predict(E_values_reshaped)
R2_E = r2_score(SRQ_median_E, SRQ_pred_E)

# Régression linéaire pour ν
nu_values_reshaped = nu_values.reshape(-1, 1)
model_nu = LinearRegression().fit(nu_values_reshaped, SRQ_median_nu)
SRQ_pred_nu = model_nu.predict(nu_values_reshaped)
R2_nu = r2_score(SRQ_median_nu, SRQ_pred_nu)

# Création des subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot de SRQ vs E
ax[0].scatter(E_values, SRQ_median_E, color='b', marker='o', label=f"$R^2 = {R2_E:.3f}$")
ax[0].set_xlabel("E")
ax[0].set_ylabel("SRQ médian")
ax[0].set_title("SRQ en fonction de E")
ax[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax[0].legend()

# Scatter plot de SRQ vs ν
ax[1].scatter(nu_values, SRQ_median_nu, color='r', marker='s', label=f"$R^2 = {R2_nu:.3f}$")
ax[1].set_xlabel("ν")
ax[1].set_ylabel("SRQ médian")
ax[1].set_title("SRQ en fonction de ν")
ax[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax[1].legend()

plt.tight_layout()
plt.show()

# Affichage des R²
print(f"R² pour SRQ vs E : {R2_E:.3f}")
print(f"R² pour SRQ vs ν : {R2_nu:.3f}")

import numpy as np

# Données SRQ extraites de ton tableau (les mêmes que précédemment)
SRQ_data = np.array([
    [0.048504, 0.048364, 0.04823, 0.048415, 0.048484],
    [0.052358, 0.052207, 0.052062, 0.052262, 0.052336],
    [0.06542, 0.065231, 0.065051, 0.0653, 0.065393],
    [0.041312, 0.041193, 0.041079, 0.041236, 0.041295],
    [0.05893, 0.05876, 0.058597, 0.058822, 0.058905],
    [0.038253, 0.038143, 0.038037, 0.038183, 0.038237],
    [0.043748, 0.043622, 0.043501, 0.043668, 0.04373],
    [0.046496, 0.046362, 0.046234, 0.046411, 0.046477],
    [0.035539, 0.035436, 0.035338, 0.035474, 0.035524],
    [0.071328, 0.071122, 0.070925, 0.071197, 0.071298]
])

# Calcul de la moyenne (S_bar) pour chaque colonne de SRQ_data (chaque vecteur nu)
S_bar = np.mean(SRQ_data, axis=0)

# Calcul de l'incertitude (u_input) pour chaque colonne de SRQ_data
n = len(SRQ_data)  # Nombre de valeurs pour chaque nu

u_input = np.zeros(5)  # Tableau pour les incertitudes

for i in range(5):
    # Calcul de la variance
    variance = np.sum((SRQ_data[:, i] - S_bar[i])**2) / (n - 1)
    # L'écart-type est la racine carrée de la variance
    u_input[i] = np.sqrt(variance)

# Affichage des résultats
for i in range(5):
    print(f"Incertitude pour ν = {nu_values[i]:.8f} : u_input = {u_input[i]:.6f}")
