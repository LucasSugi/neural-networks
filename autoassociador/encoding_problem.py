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
def mlp_architecture(input_data,output_data):
    
    input_length = input_data.shape[0]*input_data.shape[1]
    hidden_length = int(np.log(input_length)/np.log(2))
    
    #Create and populate the weigths of hidden layer
    hidden_weights = np.zeros([hidden_length,input_length+1])
    for i in range(hidden_length):  
        for j in range(input_length+1):
            hidden_weights[i][j] = rd.uniform(-0.5,0.5)

    
    output_length = output_data.shape[0] * output_data.shape[1]

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
def mlp_backward(input_data,output_data,hidden_weights,output_weights,eta):
    
    #Conditions of stop
    threshold = 0.001
    sqerror = 2 * threshold
    while(sqerror > threshold):
        sqerror = 0
        
        #Forward
        net_h,f_h,net_o,f_o = mlp_forward(input_data,hidden_weights,output_weights)
    
        #Calculates the error
        error = output_data - f_o

        #Squared error
        sqerror =  sqerror + np.sum(np.power(error,2))

        #Backpropagation
        delta_o = np.multiply(error,derivative_function(f_o))
        w_o = output_weights[:,0:output_weights.shape[1]-1]
        delta_h = np.multiply(derivative_function(f_h),np.dot(delta_o.reshape(1,f_o.shape[0]),w_o))
    
        #Learning
        output_weights =  output_weights + (eta * np.dot(delta_o.reshape(f_o.shape[0],1),np.append(f_h,1).reshape(1,f_h.shape[0]+1)))
        hidden_weights = hidden_weights + (eta * np.dot(delta_h.reshape(f_h.shape[0],1),input_data.reshape(1,input_data.shape[0])))

        #print(sqerror)

    return hidden_weights,output_weights

#Apply the mlp
def mlp(input_data,output_data,hidden_weights,output_weights):
    
    #Convert to a vector - input
    row,col = input_data.shape
    input_data = input_data.reshape(row*col,)
    
    #Convert to a vector - output
    row,col = output_data.shape
    output_data = output_data.reshape(row*col,)
    
    #Append the theta
    input_data = np.append(input_data,1)
    
    #Training
    hidden_weights,output_weights = mlp_backward(input_data,output_data,hidden_weights,output_weights,0.1)
    
    #Test
    net_h,f_h,net_o,f_o = mlp_forward(input_data,hidden_weights,output_weights)
    print(np.round(output_data-f_o))

#Read the size of input/output matrix
size_input = int(input())
size_output = int(input())

#Create the identity matrix to be coded or decoded
input_data = np.identity(size_input)
output_data = np.identity(size_output)

#MLP - Architecture
hidden_weights, output_weights = mlp_architecture(input_data,output_data);

#Mlp
mlp(input_data,output_data,hidden_weights,output_weights)
