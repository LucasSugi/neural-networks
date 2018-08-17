'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: Recognize A and A inverted with Adaline
'''

import numpy as np
import random

#Read data from file
def readData():

	#Especify the path of data
	pathData = str(input()).rstrip()
	
	#Especify the size of matrix
	sizeMatrix = int(input())
	
	#Especify the number of examples
	numberExamples = int(input())
	
	#Read datafile
	datafile = open(pathData,"r")
	matrixDatafile = datafile.read()
	
	#Close datafile	
	datafile.close()
	
	#Create numpy matrix that will hold the examples
	data = np.zeros([numberExamples,(sizeMatrix*sizeMatrix)+1],dtype=np.int32)
	
	#Position of first class
	startClass = 0
	
	#Store data from datafile to data	
	for i in range(numberExamples):
		#Get class
		data[i,(sizeMatrix*sizeMatrix)] = int(matrixDatafile[startClass:startClass+2])
		
		#Get letter
		startIndex = startClass+3
		endIndex = startIndex+(sizeMatrix*2)
		for j in range(sizeMatrix):
			row = matrixDatafile[startIndex+((sizeMatrix*2)+1)*j:endIndex+((sizeMatrix*2)+1)*j]
			for k in range(sizeMatrix):
				data[i,(j*sizeMatrix)+k] = int(row[(k*2):(k*2)+2])
		
		startClass += ((sizeMatrix*sizeMatrix)*2) + sizeMatrix+4
	
	return data

#Activation function for create 2 regions
def activationFunction(net):
	if(net >= 0.5): return 1
	else: return -1


#Train of adaline
def train(data):
	
	#Generate the weights matrix	
	weights = np.zeros(data.shape[1])
	
	#Populate matrix
	for i in range(weights.shape[0]):
		weights[i] = random.uniform(-0.5,0.5)
	
	#Classes of letters
	classIndex = data.shape[1]-1

	#Number of examples
	numberExamples = data.shape[0]	
	
	#Eta
	eta = 0.1
	
	#train model
	threshold = 0
	sqerror = 2
	while(sqerror > threshold):
		sqerror = 0
		for j in range(numberExamples):
			#copy data and set 1 for theta
			tempData = np.copy(data[j,])
			tempData[classIndex] = 1
			
			#net of multiply
			net = np.sum(np.multiply(weights,tempData))

			#result
			result = activationFunction(net)

			#Error
			error = data[j,classIndex] - result
			
			#Square error	
			sqerror = sqerror + (error*error)

			#update of weights
			if(error != 0):
				dE2 = error*(-1)*(tempData)
				weights = weights - eta * dE2

		#mean error
		sqerror = sqerror / numberExamples

	return weights

#Test of adaline
def test(weights,data):
	
	#Classes of letters
	classIndex = data.shape[1]-1

	#test the model
	accuracy = 0
	for i in range(data.shape[0]):
		#copy data and set 1 for theta
		tempData = np.copy(data[i,])
		tempData[classIndex] = 1

		#net of multiply
		net = np.sum(np.multiply(weights,tempData))

		#result
		result = activationFunction(net)

		#accuracy
		if(data[i,classIndex] - result == 0):
			accuracy += 1

	print(accuracy/data.shape[0])

#Adaline's algorithm
def adaline():
	
	#Read content from files train and test
	dataTrain = readData()
	dataTest = readData()
	
	#Test the model several times	
	numberTests = 1000
	for i in range(numberTests):
		#train a model
		weights = train(dataTrain)

		#test model
		test(weights,dataTest)

#Call for function
adaline();
