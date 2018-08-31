'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: Mlp for classification
'''

import numpy as np
import random as rd

#Activation function - Sigmoid
def activaction_function(net):
    return 1/(1+(np.exp(-net)))

#Derivative function - Sigmoid
def derivative_function(f_net):
    return f_net * (1-f_net)

#Multilayer-Perceptron Architecture
def mlp_architecture(data,hidden_length,output_length):

    #Size data
    row,col = data.shape
    
    #Create and populate the weigths of hidden layer
    hidden_weights = np.zeros([hidden_length,col])
    for i in range(hidden_length):  
        for j in range(col):
            hidden_weights[i][j] = rd.uniform(-0.5,0.5)


    #Create and populate the weigths of output layer
    output_weights = np.zeros([output_length,hidden_length+1])
    for i in range(output_length):
        for j in range(hidden_length+1):
            output_weights[i][j] = rd.uniform(-0.5,0.5)

    return hidden_weights,output_weights

#Multilayer-Perceptron Forward
def mlp_forward(tuple_data,hidden_weights,output_weights):
    
    #Computes the net and f(net) - Hidden
    net_h = np.dot(hidden_weights,tuple_data)
    f_h = activaction_function(net_h)
        
    #Set the theta
    temp_f_h = np.append(f_h,1)
    
    #Computes the net and f(net) - Output
    net_o = np.dot(output_weights,temp_f_h)
    f_o = activaction_function(net_o)
        
    return net_h,f_h,net_o,f_o


#Multilayer-Perceptron Backward
def mlp_backward(data,hidden_weights,output_weights,eta):

    #Extract class
    classes = np.copy(data[:,data.shape[1]-1])
    
    #Extract the attributes
    attributes = np.copy(data[:,0:data.shape[1]-1])
    
    #Append the theta
    theta = np.ones([data.shape[0],1])
    attributes = np.append(attributes,theta,axis=1)
   
    #Conditions of stop
    threshold = 0.01
    sqerror = 2 * threshold
    while(sqerror > threshold):
        sqerror = 0

        #For each row apply the forward and backpropagation
        row = attributes.shape[0]
        for i in range(row):
            net_h,f_h,net_o,f_o = mlp_forward(attributes[i,:],hidden_weights,output_weights)
    
            #Calculates the error
            error = classes[i] - f_o
            
            #Squared error
            sqerror =  sqerror + np.sum(np.power(error,2))

            #Backpropagation
            delta_o = np.multiply(error,derivative_function(f_o))
            w_o = output_weights[:,0:output_weights.shape[1]-1]
            delta_h = np.multiply(derivative_function(f_h),np.dot(delta_o.reshape(1,f_o.shape[0]),w_o))
    
            #Learning
            output_weights =  output_weights + (eta * np.dot(delta_o.reshape(f_o.shape[0],1),np.append(f_h,1).reshape(1,f_h.shape[0]+1)))
            hidden_weights = hidden_weights + (eta * np.dot(delta_h.reshape(f_h.shape[0],1),attributes[i,:].reshape(1,data.shape[1])))

        sqerror = sqerror / row 
        print(sqerror)

    return hidden_weights,output_weights

#Normalizing
def normalizing(data):
    
    #Size data
    row,col = data.shape

    for i in range(1,col):
        data[:,i] = (data[:,i] - min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
    return data

#Read the data that will be used in ml
def readData():
    
    #Read the name or path of data
    pathData = str(input()).rstrip()

    #Number of neurons hidden layer
    hidden_length = int(input())

    #Number of neurons output layer
    output_length = int(input())

    #Load data
    return np.genfromtxt(pathData, delimiter=","),hidden_length,output_length

#Apply the mlp
def mlp(data,hidden_weights,output_weights):
    
    #training
    hidden_weights,output_weights = mlp_backward(data,hidden_weights,output_weights,0.1)

    #test
    for i in range(data.shape[0]):
        net_h,f_h,net_o,f_o = mlp_forward(data[i,:],hidden_weights,output_weights)
        print(np.round(f_o))

#Call for read data
data,hidden_length,output_length = readData();

#MLP - Architecture
hidden_weights, output_weights = mlp_architecture(data,hidden_length,output_length);

#Normalizing data
#data = normalizing(data)

#Mlp
mlp(data,hidden_weights,output_weights)
