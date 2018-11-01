'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: PCA 
'''

import numpy as np

class PCA:
    def __init__(self,k=None):
        self.k = k
        self.explained_variance_ratio_ = []
    
    def fit_transform(self,X):
        #Convert to numpy array
        features = np.array(X)

        #Correlation matrix
        cor_features = np.corrcoef(features.T)
    
        #Eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(cor_features)
        
        #Computes the percentage of each eig_vals
        tot = np.sum(eig_vals)
        self.explained_variance_ratio_ = [i/tot for i in eig_vals]
    
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(reverse=True)
        
        #Check if has the k top vectors to choose
        if(self.k == None):
            self.k = len(eig_pairs)
        
        #Create matrix w
        w = np.array([]).reshape(len(eig_pairs),0)
        for i in range(self.k):
            w = np.concatenate((w,eig_pairs[i][1].reshape(len(eig_pairs),1)),axis=1)
        
        #Project data
        return features.dot(w)
