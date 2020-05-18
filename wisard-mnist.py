# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
CPS841 - Redes Neurais Sem Peso
Aplicação do classificador WiSARD na base de dados MNIST
Aluno: Cleiton Moya de Almeida

A implementar:
    - testar o melhor n_i entre {2, 4, 7, 8, 14, 16}, 28, 
    - criar uma função que, uma vez treinada a rede, retorna a classificaão
      de um exemplar aleatório, retornando a medida de similaridade de cada
      classe, bem como o a confiança
    
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import wisardpkg as wp


# ---------------------------- FUNÇÕES AUXILIARES ----------------------------


def plotaExemplaresConfusao(X,Y,G,y,g):
    # data-frame com todas as classificações
    df = pd.DataFrame({'Y':Y,'G':G,'X':X})
    
    # classificações incorretas da classe 'c'
    dfC = df.loc[(df['Y']==y) & (df['G']==g)]
    
    # sorteia 10 exemplares mal-classificados
    n, _ = dfC.shape

    if n == 0:
        print("Não há confusão para o 'y' e 'g' informado")
    else:
        if n<10: 
            k = n 
        else: 
            k = 10
            
        pos = random.sample(range(n),k)
        dfR = dfC.iloc[pos]
        R =  dfR['X'].to_numpy()
    
        # plota os exemplares mal-classificados
        for k,arr in enumerate(R):
            arr = np.array(arr)
            plt.subplot(2,5,k+1)
            plt.imshow(arr.reshape(28,28), cmap='gray',vmin=0, vmax=1)
            plt.axis('off')
            plt.title(dfR.iloc[k,1])
    

# ---------------------------- PARÂMETROS GERAIS ----------------------------

threshold = 128     # limite para a transfomação grayscale -> p&b
selNdados = False   # aplica somente sem em um sub-conjunto de dados
N1 = 100            # número de dados de treinamento
N2 = 10             # número de dados de teste
N = 1              # número de experimentos

# Parâmetros da WiSARD
addressSize = 14     # número de bits de enderaçamento das RAMs
ignoreZero  = False  # RAMs ignoram o endereço 0
verbose = False
returnActivationDegree = False # retorna o grau de similariedade de cada sapida
returnConfidence = False       # retorna o grau de confiança de cada saída
returnClassesDegrees = False   # optional


# -------------------- LEITURA E PRÉ-TRATAMENTO DOS DADOS --------------------

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
def step(x):
    return 1 * (x > 0)

D_tr = D_tr.apply(lambda x: step(x - threshold))
D_te = D_te.apply(lambda x: step(x - threshold))
X_tr = D_tr.values.tolist()
X_te = D_te.values.tolist()

# 4. Transforma os rótulos em uma lista de strings (formato wisardpkg)
Y_tr = list(map(str,D_tr.index.to_list()))
Y_te = list(map(str,D_te.index.to_list()))



# --------------------------- APLICAÇÃO DA WiSARD --------------------------- 

Acc = np.array([]) # matriz de acurácia
for n in range(N):
    
    print("Experimento {0:1d}:".format(n))
    
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
    print("\tTreinamento concluído em {0:1.0f}s".format(endTime-startTime))
    
    # Classificação
    startTime = time.time()
    G = wsd.classify(X_te)
    endTime = time.time() 
    print("\tClassificação concluída em {0:1.0f}s".format(endTime-startTime))

# ------------------------------- AVALIAÇÕES  ------------------------------

    # Transformação das lists em arrays
    Y = np.array(list(map(int,Y_te)))
    G = np.array(list(map(int, G)))

    # Avalição da acurácia no experimento n 
    Acc = np.append(Acc, accuracy_score(Y,G))

# Acurácia média
Acc_mean = Acc.mean()
Acc_std = Acc.std()
print("\nAcurácia média: {0:1.2f} \u00B1 {1:1.2f}".format(Acc.mean(), Acc.std()))

# Matriz de confusão do último experimento
# Linhas: Y
# Colunas: G
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
C = confusion_matrix(Y,G,labels)
print("\nMatriz de confusão do último experimento:")
print(C)

plotaExemplaresConfusao(X_te,Y,G,3,8)