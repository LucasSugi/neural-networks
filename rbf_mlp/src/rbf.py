'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: Rbf for classification
'''
import numpy as np
import random as rd

#Activation function - Gaussian
def activaction_function(net):
    return np.exp(-np.sum(np.power(net,2),axis=1)*3)

#Multilayer-Perceptron Architecture
def architecture(data,input_length,hidden_length,output_length):
    
    #Create and populate the weigths - Hidden
    centroids = np.zeros([hidden_length,input_length])
    counter = input_length
    for i in range(hidden_length):  
        tmp_data = data[data[:,counter] == 1]
        for j in range(input_length):
            centroids[i,j] = np.mean(tmp_data[:,j])
        counter += 1
    
    #Create and populate the weigths - Output
    output_weights = np.zeros([output_length,hidden_length+1])
    for i in range(output_length):  
        for j in range(hidden_length+1):
            output_weights[i][j] = rd.uniform(-0.5,0.5)
            
    return centroids,output_weights

#Multilayer-Perceptron Forward
def forward(tuple_data,centroids,output_weights):
    
    #Computes the net and f(net) - Hidden
    net_h = centroids-tuple_data
    f_h = activaction_function(net_h)
    
    #Set the theta
    temp_f_h = np.append(f_h,1)
    
    #Computes the net and f(net) - Output
    f_o = np.dot(output_weights,temp_f_h)
        
    return net_h,f_h,f_o

#Multilayer-Perceptron Backward
def backward(data,input_length,centroids,output_weights,iteration,eta,momentum):
    
    #Extract class
    classes = np.copy(data[:,input_length:data.shape[1]])
    
    #Extract the attributes
    attributes = np.copy(data[:,0:input_length])
    
    #Conditions of stop
    momentum_o = 0
    for i in range(iteration):
        #For each row apply the forward and backpropagation
        row = attributes.shape[0]
        for j in range(row):
            net_h,f_h,f_o = forward(attributes[j,:],centroids,output_weights)
    
            #Calculates the error
            error = classes[j,:] - f_o
                 
            #Learning
            tmp_o = (eta * np.dot(error.reshape(output_weights.shape[0],1),np.append(f_h,1).reshape(1,output_weights.shape[1])))
            output_weights += tmp_o
            output_weights += (momentum * momentum_o)

            #Calculates momentum
            momentum_o = tmp_o

    return output_weights

#Test of mlp
def test(data,input_length,centroids,output_weights,iteration,eta,momentum,train_size):
    
    #Divide data in train and test
    train_data = int(train_size * data.shape[0])
    train_data = np.random.choice(np.arange(0,data.shape[0]),size=train_data,replace=False)
    test_data = np.setdiff1d(np.arange(0,data.shape[0]),train_data)
    train_data = data[train_data,:]
    test_data = data[test_data,:]
    
    #MLP - Backward
    output_weights = backward(train_data,input_length,centroids,output_weights,iteration,eta,momentum)
    
    #Extract class
    classes = np.copy(test_data[:,input_length:test_data.shape[1]])
    
    #Extract the attributes
    attributes = np.copy(test_data[:,0:input_length])
    
    correct = 0
    for i in range(attributes.shape[0]):
        net_h,f_h,f_o = forward(attributes[i,:],centroids,output_weights)
        if(sum(abs(classes[i,:]-np.round(f_o)))==0):
            correct += 1
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

    #Number of iterations
    iteration = int(input())

    #Learning rate
    eta = float(input())
    
    #Momentum
    momentum = float(input())

    #Train size
    train_size = float(input())

    #Load data
    return np.genfromtxt(pathData, delimiter=","),input_length,hidden_length,output_length,iteration,eta,momentum,train_size

#Call for read data
data,input_length,hidden_length,output_length,iteration,eta,momentum,train_size = readData()

#MLP - Architecture
centroids,output_weights = architecture(data,input_length,hidden_length,output_length) 

#MLP - Test
test(data,input_length,centroids,output_weights,iteration,eta,momentum,train_size)
