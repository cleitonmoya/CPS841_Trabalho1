#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:07 2020

@author: cleiton
"""


import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import wisardpkg as wp
from sklearn.model_selection import train_test_split, KFold

# ---------------------------- PARÂMETROS GERAIS ----------------------------

# Número de experimentos
N = 1              

# Carregamento dos dados
selNdados = True   # carega somente N pontos da base de daddos
N1 = 100            # número de dados de treinamento
N2 = 50             # número de dados de teste

# Pré-tratamento dos dados
threshold = 128     # limite para a transfomação grayscale -> p&b

# Divisão dos dados
splitDados = True  # unifica as duas bases e aplica split
test_size = 0      # tamalho do conjunto de testes

# K-fold
kfold = True
nsplits = 10


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
D = D_tr.append(D_te)
if splitDados:
    X_tr, X_te, Y_tr, Y_te = train_test_split(D.values, 
                                              D.index.to_list(), 
                                              test_size = test_size, 
                                              random_state = 42,
                                              shuffle = True)
elif kfolds:
    X = D.values
    Y = D.index.to_list()
else:
    X_tr = D_tr.values.to_list()
    

# 4. formata os dados de entrada e saída em listas (formato wisarpkg)
Y_tr = list(map(str,Y_tr))
Y_te = list(map(str,Y_te))

# --------------------------- APLICAÇÃO DA WiSARD --------------------------- 

kf = KFold(n_splits=nsplits)
for tr_idx, te_idx in kf.split(X_tr,):
    Xtr = X_tr[tr_idx]
    Ytr = Y_tr[tr_idx]
    Xte = X_te[te_idx]
    Yte = Y_te[te_idx]
    print(Ytr)
    print(Yte)