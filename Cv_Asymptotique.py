# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:24:34 2025

@author: fgley
"""

import numpy as np
import matplotlib.pyplot as plt

# Données
size = np.array([0.01, 0.009, 0.008, 0.006, 0.005, 0.004, 0.003])
u_max = np.array([0.030154, 0.030255, 0.030292, 0.030431, 0.030488, 0.030555, 0.030553])

size_secondary = np.array([0.008, 0.006, 0.005, 0.004, 0.003])
u_point1 = np.array([1.46350E-02, 1.46510E-02, 1.46630E-02, 1.46780E-02, 1.46810E-02])
u_point2 = np.array([2.15080E-02, 2.16180E-02, 2.16260E-02, 2.16810E-02, 2.16610E-02])

# Premier plot : SRQ seule
plt.figure(figsize=(8, 6))
plt.plot(size, u_max, marker='o', linestyle='-', label='SRQ (u_max)')
plt.xlabel('Element Size')
plt.ylabel('SRQ (u_max)')
plt.title('Convergence de la SRQ')
plt.grid(True)
plt.legend()
plt.show()

# Deuxième plot : SRQ + deux autres points avec axes secondaires
fig, ax1 = plt.subplots(figsize=(24, 6))

# Axe principal : SRQ
ax1.plot(size_secondary, u_max[-len(size_secondary):], marker='o', linestyle='-', color='b', label='SRQ (u_max)')
ax1.set_xlabel('Element Size')
ax1.set_ylabel('SRQ (u_max)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Axe secondaire 1
ax2 = ax1.twinx()
ax2.plot(size_secondary, u_point1, marker='s', linestyle='--', color='r', label='Point 1')
ax2.set_ylabel('Déplacement Point 1', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Axe secondaire 2
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Décalage du troisième axe
ax3.plot(size_secondary, u_point2, marker='^', linestyle='-.', color='g', label='Point 2')
ax3.set_ylabel('Déplacement Point 2', color='g')
ax3.tick_params(axis='y', labelcolor='g')

plt.title('Convergence aux différents points du domaine')
plt.show()