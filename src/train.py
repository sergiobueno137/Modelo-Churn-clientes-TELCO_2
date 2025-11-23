


# Código de Entrenamiento - Modelo Churn clientes-TELCO
############################################################################

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Churn'],axis=1)
    y_train = df['Churn'].map({'Yes': 1, 'No': 0})
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    LG = LogisticRegression()
    LG.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/modelo_LG.sav'
    pickle.dump(LG, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')




# Entrenamiento completo
def main():
    read_file_csv('churn_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()