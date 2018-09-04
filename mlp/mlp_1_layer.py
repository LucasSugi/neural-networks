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
def architecture(input_length,hidden_length,output_length):
    
    #Create and populate the weigths - Hidden
    hidden_weights = np.zeros([hidden_length,input_length+1])
    for i in range(hidden_length):  
        for j in range(input_length+1):
            hidden_weights[i][j] = rd.uniform(-0.5,0.5)
            
    #Create and populate the weigths - Output
    output_weights = np.zeros([output_length,hidden_length+1])
    for i in range(output_length):  
        for j in range(hidden_length+1):
            output_weights[i][j] = rd.uniform(-0.5,0.5)
            
    return hidden_weights,output_weights

#Multilayer-Perceptron Forward
def forward(tuple_data,hidden_weights,output_weights):
    
    #Computes the net and f(net) - Hidden 1
    net_h = np.dot(hidden_weights,tuple_data)
    f_h = activaction_function(net_h)
    
    #Set the theta
    temp_f_h = np.append(f_h,1)
    
    #Computes the net and f(net) - Output
    net_o = np.dot(output_weights,temp_f_h)
    f_o = activaction_function(net_o)
        
    return net_h,f_h,net_o,f_o


#Multilayer-Perceptron Backward
def backward(data,input_length,hidden_weights,output_weights,eta,momentum):
    
    #Extract class
    classes = np.copy(data[:,input_length:data.shape[1]])
    
    #Extract the attributes
    attributes = np.copy(data[:,0:input_length])

    #Append the theta
    theta = np.ones([data.shape[0],1])
    attributes = np.append(attributes,theta,axis=1)
    
    #Conditions of stop
    threshold = 0.01
    sqerror = 2 * threshold
    momentum_o = momentum_h = 0
    while(sqerror > threshold):
        sqerror = 0

        #For each row apply the forward and backpropagation
        row = attributes.shape[0]
        for i in range(row):
            net_h,f_h,net_o,f_o = forward(attributes[i,:],hidden_weights,output_weights)
    
            #Calculates the error
            error = classes[i,:] - f_o
            
            #Squared error
            sqerror =  sqerror + np.sum(np.power(error,2))
    
            #Backpropagation
            delta_o = np.multiply(error,derivative_function(f_o))
            w_o = output_weights[:,0:output_weights.shape[1]-1]
            delta_h = np.multiply(derivative_function(f_h),np.dot(delta_o.reshape(1,f_o.shape[0]),w_o))
    
            #Learning
            output_weights += (eta * np.dot(delta_o.reshape(f_o.shape[0],1),np.append(f_h,1).reshape(1,f_h.shape[0]+1)))
            output_weights += (momentum * momentum_o)
            hidden_weights += (eta * np.dot(delta_h.reshape(f_h.shape[0],1),attributes[i,:].reshape(1,input_length+1)))
            hidden_weights += (momentum * momentum_h)
            
            #Computes momentum
            momentum_o = (eta * np.dot(delta_o.reshape(f_o.shape[0],1),np.append(f_h,1).reshape(1,f_h.shape[0]+1)))
            momentum_h = (eta * np.dot(delta_h.reshape(f_h.shape[0],1),attributes[i,:].reshape(1,input_length+1)))

        sqerror = sqerror / row 
        print(sqerror)

    return hidden_weights,output_weights

#Test of mlp
def test(data,input_length,hidden_weights,output_weights):
    
    #Extract class
    classes = np.copy(data[:,input_length:data.shape[1]])
    
    #Extract the attributes
    attributes = np.copy(data[:,0:input_length])
    
    #Append the theta
    theta = np.ones([data.shape[0],1])
    attributes = np.append(attributes,theta,axis=1)

    correct = 0
    for i in range(attributes.shape[0]):
        net_h,f_h,net_o,f_o = forward(attributes[i,:],hidden_weights,output_weights)
        if(sum(abs(classes[i]-np.round(f_o)))==0):
            correct = correct + 1

    print("Accuracy: ",correct/attributes.shape[0])

#Read the data that will be used in ml
def readData():
    
    #Read the name or path of data
    pathData = str(input()).rstrip()
    
    #Number of neurons input layer
    input_length = int(input())

    #Number of neurons hidden layer
    hidden_length = int(input())
    
    #Number of neurons output layer
    output_length = int(input())

    #Load data
    return np.genfromtxt(pathData, delimiter=","),input_length,hidden_length,output_length

#Call for read data
data,input_length,hidden_length,output_length = readData()

#MLP - Architecture
hidden_weights,output_weights = architecture(input_length,hidden_length,output_length) 

#MLP - Backward
hidden_weights,output_weights = backward(data,input_length,hidden_weights,output_weights,0.2,0.7)

test(data,input_length,hidden_weights,output_weights)
