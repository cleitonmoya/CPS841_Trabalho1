#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:43:58 2020

@author: cleiton
"""


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