#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:42:18 2020

@author: cleiton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

''' 
Conjunto de teste: MNIST
Tamanho de cada imagem: 28x28 pixels
'''

# 1. Lê os conjuntos de dados de treinamento e teste
#train = d.read_csv('dataset/mnist_train.csv', index_col="label")
test = pd.read_csv('dataset/mnist_test.csv', index_col="label")
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Sub-conjuntos de cada label
#D0 = test.loc[0]
#D1 = test.loc[1]
D2 = test.loc[2]

# 3. Realiza um pré-tratamento das imagens de forma a torná-las monocromática
D2 = D2.iloc[0:10,:]
D2 = D2.apply(lambda x: 255*np.heaviside(x-122,0))

# 3.1 Torna o vetor binário e transforma em lista de listas
# x = 0, se x <= 122; x = 1, se x > 122
D2b = D2.apply(lambda x: np.heaviside(x-122,0))
D2L = D.values.tolist()

# 3.2 Transforma os labels em uma lista de strings
y =list(map(str,D2.index.to_list()))

# Transforma o conjunto em np.array (para plotagem), excluindo o label
d2 = D2.to_numpy()

'''
idxDado = 635
arr = test.iloc[idxDado,:].to_numpy().reshape(28,28)
'''
# Plota o subconjunto de 10 exemplares
for k,arr in enumerate(d2):
    plt.subplot(2,5,k+1)
    plt.imshow(arr.reshape(28,28), cmap='gray',vmin=0, vmax=255)
    plt.axis('off')
    