

# Código de Scoring - Modelo Modelo Churn clientes - TELCO
############################################################################

import pandas as pd
import numpy as np
import os
import pickle
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report,accuracy_score,precision_score, recall_score
#from sklearn.metrics import roc_auc_score,roc_curve


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/modelo_LG.sav'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('churn_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()