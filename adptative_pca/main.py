'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: Comparison between PCA and Adaptive PCA
'''

#Import's
import pandas as pd
import numpy as np
from pca import PCA
from apca import AdaptivePCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#Read wine
wine = datasets.load_wine()
wine = pd.DataFrame(data= np.c_[wine['data'], wine['target']],columns= wine['feature_names'] + ['target'])

#Separate in features and target
features = wine.iloc[:,0:13]
target = wine.iloc[:,13]

#Standardization dataset
features = StandardScaler().fit_transform(features)

#Creating pca
pca = PCA(k=5)
newFeatures1 = pca.fit_transform(features)

#Creating adaptive pca
aPCA = AdaptivePCA(13,5,100)
newFeatures2 = aPCA.fit_transform(features)

#Split dataset into train and test
kf = KFold(n_splits=10)

#Mlp model
mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(25,))

score = {'None':[],'pca':[],'aPca':[]}

#Run for each fold
for train_index, test_index in kf.split(features):
    #Apply classification for dataset without pca
    mlp.fit(features[train_index],target[train_index])
    score['None'] = np.append(score['None'],mlp.score(features[test_index],target[test_index]))

    #Apply classification for dataset with pca
    mlp.fit(newFeatures1[train_index],target[train_index])
    score['pca'] = np.append(score['pca'],mlp.score(newFeatures1[test_index],target[test_index]))

    #Apply classification for dataset with adaptive pca
    mlp.fit(newFeatures2[train_index],target[train_index])
    score['aPca'] = np.append(score['aPca'],mlp.score(newFeatures2[test_index],target[test_index]))
    
    
#Mean of each test
print('Mean accuracy for None:',np.round(np.mean(score['None']),3))
print('Mean accuracy for pca:',np.round(np.mean(score['pca']),3))
print('Mean accuracy for aPca:',np.round(np.mean(score['aPca']),3),'\n')

#Test if mean is equal
print('P-value between None and pca:',np.round(stats.ttest_ind(score['None'],score['pca']).pvalue,3))
print('P-value between None and aPca:',np.round(stats.ttest_ind(score['None'],score['aPca']).pvalue,3))
print('P-value between pca and aPca:',np.round(stats.ttest_ind(score['pca'],score['aPca']).pvalue,3))
