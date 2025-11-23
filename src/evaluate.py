

# Código de Evaluación - Modelo Churn clientes - TELCO
############################################################################

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,precision_score, recall_score
from sklearn.metrics import roc_auc_score,roc_curve


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/modelo_LG.sav'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Churn'],axis=1)
    y_test = df['Churn'].map({'Yes': 1, 'No': 0})
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    eval_model('churn_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()





