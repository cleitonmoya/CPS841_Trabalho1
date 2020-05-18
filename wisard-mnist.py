# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
CPS841 - Redes Neurais Sem Peso
Aplicação do classificador WiSARD na base de dados MNIST
Aluno: Cleiton Moya de Almeida

A implementar:
    - veriicar se dá para fixar o mapeamento
    - testar o melhor n_i entre {2, 4, 7, 8, 14, 16}, 28, 
    - criar uma função que, uma vez treinada a rede, retorna a classificaão
      de um exemplar aleatório, retornando a medida de similaridade de cada
      classe, bem como o a confiança
    
"""
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import wisardpkg as wp

# ---------------------------- PARÂMETROS GERAIS ----------------------------

threshold = 128     # limite para a transofmração grayscale -> p&b
selNdados = False   # aplica somente sem em um sub-conjunto de dados
N1 = 100          # número de dados de treinamento
N2 = 10          # número de dados de teste

# Parâmetros da WiSARD
addressSize = 8     # number of addressing bits in the ram
ignoreZero  = False  # causes the rams to ignore the address 0
verbose = False
returnActivationDegree = False # optional
returnConfidence = False       # optional
returnClassesDegrees = False    # optional

# -------------------- LEITURA E PRÉ-TRATAMENTO DOS DADOS --------------------

def step(x):
    return 1 * (x > 0)

# 1. Lê os conjuntos de dados de treinamento e teste
#    train_df.sahpe = (60000, 784) / teste_df.shape = (10000, 784) 
D_tr = pd.read_csv('dataset/mnist_train.csv', index_col="label")
D_te = pd.read_csv('dataset/mnist_test.csv', index_col="label")

# 2. Seleciona N1 dados de treinamento e N2 dados de teste 
if selNdados:
    D_tr = D_tr.iloc[0:N1,:]
    D_te = D_te.iloc[0:N2,:]

# 3. Torna as imagens mono-cromáticas e formata os dados de entrada
#    X em listas binárias
D_tr = D_tr.apply(lambda x: step(x - threshold))
D_te = D_te.apply(lambda x: step(x - threshold))
X_tr = D_tr.values.tolist()
X_te = D_te.values.tolist()

# 4. Transforma os rótulos em uma lista de strings (formato wisardpkg)
Y_tr = list(map(str,D_tr.index.to_list()))
Y_te = list(map(str,D_te.index.to_list()))


# --------------------------- APLICAÇÃO DA WiSARD --------------------------- 

# Criação do modelo
wsd = wp.Wisard(addressSize,
                ignoreZero = ignoreZero, 
                verbose = verbose, 
                returnActivationDegree = returnActivationDegree,
                returnConfidence = returnConfidence,
                returnClassesDegrees = returnClassesDegrees)

# Treinamento
startTime = time.time()
wsd.train(X_tr,Y_tr)
endTime = time.time() 
print("Treinamento concluído em {0:1.0f}s".format(endTime-startTime))

# Classificação (hipótese G)
startTime = time.time()
G = wsd.classify(X_te)
endTime = time.time() 
print("Classificação concluída em {0:1.0f}s".format(endTime-startTime))


# ------------------------------- AVALIAÇÕES  ------------------------------

# Transformação das lists em arrays
Y = np.array(list(map(int,Y_te)))
G = np.array(list(map(int, G)))

# Avalição da acurácia
Acc = accuracy_score(Y,G)
print("Acurácia G: ", Acc)

# Matriz de confusão
C = confusion_matrix(Y,G)
print("Matriz de confusao:")
print(C)

