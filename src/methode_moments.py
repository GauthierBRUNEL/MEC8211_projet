import os
import pandas as pd
import numpy as np

print('============================')

# Obtenir le chemin absolu du script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construire le chemin absolu du fichier CSV (remonte d'un niveau, puis dans data)
file_path = os.path.join(script_dir, "..", "data", "Couples_LHS_LA.csv")
print("Chemin utilisé :", file_path)

# Lire le fichier CSV en précisant que le séparateur est la virgule
df = pd.read_csv(file_path, sep=",", header=0, encoding="ISO-8859-1")

# Afficher les colonnes pour vérifier
print("Colonnes détectées :", df.columns.tolist())

# Calcul de la moyenne et de l'écart-type à partir des colonnes "E_LA" et "nu_LA"
mu_E = df["E_LA"].mean()
sigma_E = df["E_LA"].std()
mu_nu = df["nu_LA"].mean()
sigma_nu = df["nu_LA"].std()

print(f"µ_E = {mu_E:.5f}, σ_E = {sigma_E:.5f}")
print(f"µ_ν = {mu_nu:.5f}, σ_ν = {sigma_nu:.5f}")

# Choisir un pourcentage de variation (ex: 2%)
p = 0.02  

# Calculer les valeurs perturbées
E_plus = mu_E * (1 + p)
E_minus = mu_E * (1 - p)
nu_plus = mu_nu * (1 + p)
nu_minus = mu_nu * (1 - p)

# Affichage des valeurs à utiliser dans les simulations pour les dérivées
print(f"Valeurs pour la simulation (avec {p*100}% de variation) :")
print(f"E+  = {E_plus:.5f}, E-  = {E_minus:.5f}")
print(f"nu+ = {nu_plus:.5f}, nu- = {nu_minus:.5f}")

# Le fichier u_max_methode_moments.csv contient les résultats des simulations pour u_max.
# L'ordre des simulations est le suivant :
# Ligne 0 : (E_plus, µ_nu, u_max)      --> Simulation avec E+ et nu moyen
# Ligne 1 : (E_minus, µ_nu, u_max)     --> Simulation avec E- et nu moyen
# Ligne 2 : (µ_E, nu_plus, u_max)      --> Simulation avec E moyen et nu+
# Ligne 3 : (µ_E, nu_minus, u_max)     --> Simulation avec E moyen et nu-

# Construire le chemin absolu du fichier de résultats
file_path_u = os.path.join(script_dir, "..", "data", "u_max_methode_moments.csv")

# Lire le fichier CSV des résultats
df_u = pd.read_csv(file_path_u, sep=",", header=0, encoding="ISO-8859-1")
print("Colonnes dans u_max_methode_moments.csv :", df_u.columns.tolist())
print("Aperçu des données :", df_u.head())

# Convertir les colonnes concernées en float (si nécessaire)
df_u['E'] = pd.to_numeric(df_u['E'], errors='coerce')
df_u['nu'] = pd.to_numeric(df_u['nu'], errors='coerce')
df_u['u_max'] = pd.to_numeric(df_u['u_max'], errors='coerce')

# Récupérer les résultats dans l'ordre indiqué
# Pour E :
E_plus_csv    = df_u.iloc[0]['E']   # Valeur de E pour la simulation avec E+
u_max_E_plus  = df_u.iloc[0]['u_max']

E_minus_csv   = df_u.iloc[1]['E']   # Valeur de E pour la simulation avec E-
u_max_E_minus = df_u.iloc[1]['u_max']

# Pour nu :
nu_plus_csv   = df_u.iloc[2]['nu']  # Valeur de nu pour la simulation avec nu+
u_max_nu_plus = df_u.iloc[2]['u_max']

nu_minus_csv   = df_u.iloc[3]['nu']  # Valeur de nu pour la simulation avec nu-
u_max_nu_minus = df_u.iloc[3]['u_max']

# Approximation des dérivées partielles par différences finies
# Dérivée par rapport à E :
du_dE = (u_max_E_plus - u_max_E_minus) / (E_plus_csv - E_minus_csv)
# Dérivée par rapport à nu :
du_dnu = (u_max_nu_plus - u_max_nu_minus) / (nu_plus_csv - nu_minus_csv)

print(f"d(u_max)/dE = {du_dE:.5f}")
print(f"d(u_max)/dnu = {du_dnu:.5f}")

# Application de la formule de propagation des incertitudes :
# u_input^2 = [ (∂u_max/∂E) * σ_E ]^2 + [ (∂u_max/∂nu) * σ_nu ]^2
u_input = np.sqrt((du_dE * sigma_E)**2 + (du_dnu * sigma_nu)**2)
print(f"u_input = {u_input:.5f}")