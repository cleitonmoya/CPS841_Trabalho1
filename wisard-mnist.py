# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
CPS841 - Redes Neurais Sem Peso
Aplicação do classificador WiSARD na base de dados MNIST
Aluno: Cleiton Moya de Almeida

Resultados
    - teste ni={4, 7, 14, 28, 49, 56}
        4: 0.72 +- 0.01 | 49s
        7: 0.77 +- 0.01 | 2s/10s
        14: 0.83 +- 0.01 | 1.86s/2s
    **  28: 0.91 +- 0.00 | 1.72s/1s | Sem bleaching: 0.81 | ingnoringZero: 0.83
        35: 0.92 +- 0.00 | 1.37s/0.83s
        40: 0.92 +- 0.00 | 1.50s/0.83s 
        49: 0.91 +- 0.00 | 1.55s/0.71s
        56: 0.88 +- 0.00 | 0.54s/0.66s  
        
    - teste com split
        70/30: mesmo resultado
        80/20: mesmo resultado
        90/10: mesmo resultado
        
    - teste 10-fold
    
    
"""
import numpy as np
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix 
import wisardpkg as wp

# ---------------------------- PARÂMETROS GERAIS ----------------------------

# Número de experimentos
N = 1              

# Carregamento dos dados
selNdados = False   # carega somente N pontos da base de daddos
N1 = 100            # número de dados de treinamento
N2 = 50             # número de dados de teste

# Pré-tratamento dos dados
threshold = 128     # limite para a transfomação grayscale -> p&b

# Divisão dos dados
splitDados = False  # unifica as duas bases e aplica split
test_size = 0.2    # tamalho do conjunto de testes

# K-fold
kfold = True
nsplits = 10

# Parâmetros da WiSARD
addressSize = 28                # número de bits de enderaçamento das RAMs
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

# 3. Junta e re-divide os conjuntos de dados    
if splitDados:
    D = D_tr.append(D_te)
    X_tr, X_te, Y_tr, Y_te = train_test_split(D.values.tolist(), 
                                              D.index.to_list(), 
                                              test_size = test_size, 
                                              random_state = 42,
                                              shuffle = True)
else:
    X_tr = D_tr.values.tolist()
    X_te = D_te.values.tolist()
    Y_tr = D_tr.index.to_list()
    Y_te = D_te.index.to_list()

# 4. formata os dados de saída em formato de lista de string (formato wisarpkg)
Y_tr = list(map(str,Y_tr))
Y_te = list(map(str,Y_te))

# --------------------------- APLICAÇÃO DA WiSARD --------------------------- 

Acc = np.array([])  # matriz de acurácia
T_tr = np.array([]) # matriz de tempos de treinamento
T_te = np.array([]) # matriz de tempos de classificação


for n in range(N): # caso não usar k-fold
    print("Experimento {0:1d}:".format(n))
    
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

    # Transformação das listas de string em listas de inteiros 
    G = list(map(int, G))
    Y = list(map(int, Y_te))
    
    # Avalição da acurácia no experimento n
    Acc_n = accuracy_score(Y,G)
    Acc = np.append(Acc, Acc_n)
    print("\tAcurácia {0:1.2f}".format(Acc_n))

# Tempos médios de treinamento e classificação
print("\nTempo médio de treinamento: {0:1.2f}s".format(T_tr.mean()))
print("Tempo médio de classificação: {0:1.2f}s".format(T_te.mean()))

# Acurácia média
Acc_mean = Acc.mean()
Acc_std = Acc.std()
print("\nAcurácia média: {0:1.2f} \u00B1 {1:1.2f}".format(Acc.mean(), Acc.std()))

# Matriz de confusão do último experimento
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
C = confusion_matrix(Y,G,labels)
print("\nMatriz de confusão do último experimento:")
print("\t Linhas: y")
print("\tColunas: g")
print(C)
