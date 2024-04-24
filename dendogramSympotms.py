import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
df = pd.read_csv('../Dataset_sintomas_final.csv')

X = df.drop(['Severity','participant_id'], axis=1)

# Realizar el agrupamiento jerárquico
linkage_matrix = linkage(X, method='ward')

# Crear el dendrograma
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8.)
plt.title("Dendrograma Jerárquico")
plt.xlabel("Índices de Muestras")
plt.ylabel("Distancia Euclidiana")

plt.show()