import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Cargar el conjunto de datos Iris
df = pd.read_csv('../Dataset_secuelas_final.csv')

X = df.drop(['Severity','participant_id'], axis=1)

# Realizar el agrupamiento jerárquico
linkage_matrix = linkage(X, method='ward')

# Crear el dendrograma
dendrogram(linkage_matrix,  p=5,truncate_mode='level', leaf_rotation=90., leaf_font_size=8.)
plt.title("Dendrograma Jerárquico")
plt.xlabel("Índices de Muestras")
plt.ylabel("Distancia Euclidiana")

plt.show()