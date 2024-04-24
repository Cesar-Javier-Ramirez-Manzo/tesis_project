import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv('../Dataset_sintomas_final.csv')

X = df.drop(['Severity','participant_id'], axis=1) 
y = df['Severity']  # y contiene la columna de clases

# Divide el conjunto de datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="gini")

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar el árbol de decisión (puede requerir Graphviz)

f_list = list(pd.DataFrame(X).columns)
c_list = list(pd.DataFrame(y).columns)

#plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=f_list, class_names=['0','1','2'])
plt.savefig(fname='arbolSymptomsGini',dpi=1000)

selected_variables = X[['Fever', 'Cough', 'Bodyache','Vomiting']]

# t=scatter_matrix(selected_variables, alpha=0.8, figsize=(10, 10))
# plt.show()