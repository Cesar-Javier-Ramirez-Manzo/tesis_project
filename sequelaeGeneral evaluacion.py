import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn import model_selection
import numpy as np
from sklearn.preprocessing import label_binarize

df = pd.read_csv('C:/Users/macet/Documents/Maestria/2do semestre/Python/MLfromscratch/Dataset_secuelas_final.csv')

X = df.drop(['Severity','participant_id'], axis=1) 
y = df['Severity']  # y contiene la columna de clases

# Desarrollando KFolds
kf = model_selection.KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Entrenar el modelo de árbol de decisión en cada split
    DTmodel = DecisionTreeClassifier(criterion="gini")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
    DTmodel, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)
)

    plt.figure()
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Puntaje")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Validación")
    plt.legend(loc="best")

    plt.show()
    
    DTmodel.fit(X_train, y_train)
    
    # Evaluar el modelo en el split actual
    accuracy = DTmodel.score(X_test, y_test)
    print(f"  Accuracy: {accuracy}")

    y_pred = DTmodel.predict(X_test)
    # Evaluar el rendimiento del modelo
    print("Exactitud del modelo DT:", accuracy_score(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    #Normalized confusion matrix for the DT DTmodel
    prediction_labels = DTmodel.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, prediction_labels,normalize=True)
    plt.show()
# Probabilidades predichas
    y_prob = DTmodel.predict_proba(X_test)
    
    # Binariza las etiquetas
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    
    # Calcula la curva ROC y el área bajo la curva para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(len(np.unique(y))):
        fpr[j], tpr[j], _ = roc_curve(y_test_bin[:, j], y_prob[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Graficar la curva ROC que contenga las tres clases
    plt.figure()
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f) - Class 0' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='blue', lw=2, label='ROC curve (area = %0.2f) - Class 1' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='green', lw=2, label='ROC curve (area = %0.2f) - Class 2' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Fold {i}')
    plt.legend(loc="lower right")
    plt.show()

