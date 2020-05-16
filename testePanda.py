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

# Lê os conjuntos de dados de treinamento e teste
#train = d.read_csv('dataset/mnist_train.csv', index_col="label")
test = pd.read_csv('dataset/mnist_test.csv', index_col="label")



# Transforma um exemplar em np.array
idxDado = 635
arr = test.iloc[idxDado,:].to_numpy().reshape(28,28)

# Imprime a imagem
plt.imshow(arr, cmap='gray',vmin=0, vmax=255)
plt.title(test.iloc[idxDado,:].name)

