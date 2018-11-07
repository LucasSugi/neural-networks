'''
Author: Lucas Yudi Sugi - 9293251
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kohonen import KOHONEN

#Read wine
wine = datasets.load_wine()
wine = pd.DataFrame(data= np.c_[wine['data'], wine['target']],columns= wine['feature_names'] + ['target'])

#Separate in features and target
features = wine.iloc[:,0:13]

#Neuron's grid
grid = [10,10]

#Apply kohonen
khn = KOHONEN(data=features,grid=grid,alpha=0.5)

#Train kohonen
#khn.train(100,verbose=True)

#Load the last model
khn.load('model.npy')

#Show result
khn.show(wine.iloc[:,13])
