'''
Author: Lucas Yudi Sugi - 9293251
'''

import numpy as np

class AdaptivePCA:
    def __init__(self,input_neuron,output_neuron,iteration,eta_w=0.001,eta_u=0.001,eta_a=0.0001):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.iteration = iteration
        self.w = []
        self.u = []
        self.eta_w = eta_w
        self.eta_u = eta_u
        self.eta_a = eta_a
        
    #Create the two matrix w and u
    def architecture(self):
        self.w = np.random.rand(self.input_neuron,self.output_neuron)
        self.u = np.zeros([self.output_neuron-1,self.output_neuron-1])
        for i in range(self.u.shape[0]):
            for j in range(i+1):
                self.u[i,j] = np.random.rand()
    
    #Fit and transform the model
    def fit_transform(self,X):
        self.architecture()
        for i in range(self.iteration):
            for p in X:
                #Computes the output y
                y = np.dot(p.reshape(1,self.w.shape[0]),self.w)
                y[0,1:y.shape[1]] += np.dot(self.u,y[0,0:y.shape[1]-1])
                
                #Update matrix w                 
                self.w +=  (self.eta_w*np.dot(p.reshape(self.w.shape[0],1),y)) - (self.eta_a * np.dot(self.w,np.power(y,2).reshape(self.w.shape[1],1)))
                
                #Norm vector
                self.w = np.divide(self.w,np.linalg.norm(self.w,axis=0))
            
                #Update matrix u
                for j in range(self.u.shape[0]):
                    for k in range(j+1):
                        self.u[j,k] += -(self.eta_u * y[0,j+1] * y[0,k])
        return X.dot(self.w)
