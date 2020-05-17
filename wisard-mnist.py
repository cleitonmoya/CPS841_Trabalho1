# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
CPS841 - Redes Neurais Sem Peso
Aplicação do classificador WiSARD na base de dados MNIST
Aluno: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import wisardpkg as wp

# ---------------------------- PARÂMETROS GERAIS ----------------------------

threshold = 128     # limite para a transofmração grayscale -> p&b
N1 = 100
N2 = 50

# Parâmetros da WiSARD
addressSize = 3     # number of addressing bits in the ram
ignoreZero  = False # causes the rams to ignore the address 0
verbose = False

# -------------------- LEITURA E PRÉ-TRATAMENTO DOS DADOS --------------------

def step(x):
    return 1 * (x > 0)

# 1. Lê os conjuntos de dados de treinamento e teste
trein_df = pd.read_csv('dataset/mnist_train.csv', index_col="label")
teste_df = pd.read_csv('dataset/mnist_test.csv', index_col="label")

# 2. Seleciona N1 dados de treinamento e N2 dados de teste 
D_tr = trein_df.iloc[0:N1,:]
D_te = teste_df.iloc[0:N2,:]

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
wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

# Treinamento
wsd.train(X_tr,Y_tr)

# Classificação (hipótese G)
G = wsd.classify(X_te)

# ---------------------------- AVALIAÇÃO DO ERRO ---------------------------
# Transformação das lists em arrays
Y = np.array(list(map(int,Y_te)))
G = np.array(list(map(int, G)))

# Avalição do erro
Ein = (G!=Y).mean()

print("Erro: ", Ein)