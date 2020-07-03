# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
CPS841 - Redes Neurais Sem Peso
Aplicação do classificador WiSARD na base de dados MNIST
Aluno: Cleiton Moya de Almeida
    
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import  KFold
from sklearn.metrics import accuracy_score, confusion_matrix 
import wisardpkg as wp

# ---------------------------- PARÂMETROS GERAIS ----------------------------
       
# Carregamento dos dados
selNdados = False   # carega somente N pontos da base de daddos
N1 = 100            # número de dados de treinamento
N2 = 50             # número de dados de teste

# Pré-tratamento dos dados
threshold = 64     # limite para a transfomação grayscale -> p&b

# K-fold
nsplits = 10

# Parâmetros da WiSARD
addressSize = 14                # número de bits de enderaçamento das RAMs
bleachingActivated= True        # desempate
ignoreZero  = False             # RAMs ignoram o endereço 0
completeAddressing = True       # quando M (núm. bits) não é divisível por n_i
verbose = False                 # mensanges durante a execução
returnActivationDegree = False  # retornq o grau de similariedade de cada y
returnConfidence = False        # retorna o grau de confiança de cada y
returnClassesDegrees = False    # confiança de cada y em relação a cada classe


# -------------------- LEITURA E PRÉ-TRATAMENTO DOS DADOS --------------------

# 1. Lê e divide os conjuntos de dados de treinamento e teste
#    train_df.sahpe = (60000, 784) / teste_df.shape = (10000, 784) 
if selNdados:
    D_tr = pd.read_csv('dataset/mnist_train.csv', index_col="label", nrows=N1)
    D_te = pd.read_csv('dataset/mnist_test.csv', index_col="label", nrows=N2)
else:
    D_tr = pd.read_csv('dataset/mnist_train.csv', index_col="label")
    D_te = pd.read_csv('dataset/mnist_test.csv', index_col="label")
    
# 2. Torna as imagens mono-cromáticas (pré-processamento)
def step(x):
    return 1 * (x > 0)
D_tr = D_tr.apply(lambda x: step(x - threshold))
D_te = D_te.apply(lambda x: step(x - threshold))

# 3. Prepara os dados para o k-fold (np.array) 
D = D_tr.append(D_te)
X = D.values
Y = D.index.to_numpy()

# 4. formata os dados de saída em np.array de string (formato wisarpkg)
Y = np.array(list(map(str,Y)))


# --------------------------- APLICAÇÃO DA WiSARD --------------------------- 

Acc = np.array([])  # matriz de acurácia
T_tr = np.array([]) # matriz de tempos de treinamento
T_te = np.array([]) # matriz de tempos de classificação

kf = KFold(n_splits=nsplits) # modelo kf
k=1
for tr_idx, te_idx in kf.split(X):
    
    print("Fold {0:1d}:".format(k))
    
    X_tr = X[tr_idx]
    X_te = X[te_idx]
    Y_tr = Y[tr_idx]
    Y_te = Y[te_idx]

    # Criação do modelo
    wsd = wp.Wisard(addressSize,
                    bleachingActivated = bleachingActivated,
                    ignoreZero = ignoreZero,
                    completeAddressing = completeAddressing,
                    verbose = verbose,
                    returnActivationDegree = returnActivationDegree,
                    returnConfidence = returnConfidence,
                    returnClassesDegrees = returnClassesDegrees)
    
    # Treinamento
    startTime = time.time()
    wsd.train(X_tr,Y_tr)
    endTime = time.time()
    T_tr_n = endTime-startTime
    print("\tTreinamento concluído em {0:1.2f}s".format(T_tr_n))
    T_tr = np.append(T_tr,T_tr_n)

    # Classificação
    startTime = time.time()
    G = wsd.classify(X_te)
    endTime = time.time()
    T_te_n = endTime-startTime
    print("\tClassificação concluída em {0:1.2f}s".format(T_te_n))
    T_te = np.append(T_te,T_te_n)

# ------------------------------- AVALIAÇÕES  ------------------------------

    # Transformação de Y,G para listas de inteiros 
    G = list(map(int, G))       # G original: lista de string
    Y_te = list(map(int, Y_te))    # Y_te original: array de string
    
    # Avalição da acurácia no experimento n
    Acc_n = accuracy_score(Y_te,G)
    Acc = np.append(Acc, Acc_n)
    print("\tAcurácia {0:1.2f}".format(Acc_n))
    
    k=k+1

# Tempos médios de treinamento e classificação
print("\nTempo médio de treinamento: {0:1.2f}s".format(T_tr.mean()))
print("Tempo médio de classificação: {0:1.2f}s".format(T_te.mean()))

# Acurácia média
Acc_mean = Acc.mean()
Acc_std = Acc.std()
print("\nAcurácia média: {0:1.2f} \u00B1 {1:1.2f}".format(Acc.mean(), Acc.std()))

# Matriz de confusão do último experimento
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
C = confusion_matrix(Y_te,G,labels)
print("\nMatriz de confusão do último experimento:")
print("\t Linhas: y")
print("\tColunas: g")
print(C)
