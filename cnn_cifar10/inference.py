'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0270 - Introducao a Redes Neurais 
Title: CNN in CIFAR-10
References: https://www.tensorflow.org/tutorials/estimators/cnn
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random as rd
import imageio as img
from cnn_cifar10 import cnn_model_fn

filter_layer_1 = 64
filter_layer_2 = 128
filter_layer_3 = 256
units_dense_1 = 512
units_dense_2 = 1024

def normalize(data):
  #Normalize data
  for i in range(data.shape[0]):
      data[i,:,:,0] = (data[i,:,:,0]-np.min(data[i,:,:,0])) / (np.max(data[i,:,:,0])-np.min(data[i,:,:,0]))
      data[i,:,:,1] = (data[i,:,:,1]-np.min(data[i,:,:,1])) / (np.max(data[i,:,:,1])-np.min(data[i,:,:,1]))
      data[i,:,:,2] = (data[i,:,:,2]-np.min(data[i,:,:,2])) / (np.max(data[i,:,:,2])-np.min(data[i,:,:,2]))
  return data

def read_image():
  #Read image
  data = np.zeros([10,32,32,3],dtype=np.float32)
  data[0,:,:,:] = img.imread('dataset/airplane.jpg')
  data[1,:,:,:] = img.imread('dataset/automobile.jpg')
  data[2,:,:,:] = img.imread('dataset/bird.jpg')
  data[3,:,:,:] = img.imread('dataset/cat.jpg')
  data[4,:,:,:] = img.imread('dataset/deer.jpg')
  data[5,:,:,:] = img.imread('dataset/dog.jpg')
  data[6,:,:,:] = img.imread('dataset/frog.jpg')
  data[7,:,:,:] = img.imread('dataset/horse.jpg')
  data[8,:,:,:] = img.imread('dataset/ship.jpg')
  data[9,:,:,:] = img.imread('dataset/truck.jpg')
  return data

def main(unused_argv):

  #Read data for inference
  data = read_image()
  labels = np.asarray([0,1,2,3,4,5,6,7,8,9],dtype=np.int32)

  #Normalize
  data = normalize(data)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='model_cifar10',warm_start_from='model_cifar10')

  #Dictionary of classes
  keys = {0:'Airplane',1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'Dog',6:'Frog',7:'Horse',8:'Ship',9:'Truck'}

  # Predict
  predict_model = tf.estimator.inputs.numpy_input_fn(x=data,num_epochs=1,shuffle=False)
  predict_result = list(cifar10_classifier.predict(input_fn=predict_model))
  counter = 0
  print()
  for i in range(len(predict_result)):
      print('True class: ',keys[labels[i]])
      print('Predict class: ',keys[predict_result[i]['classes']])
      print('Probability of the class: ',np.max(predict_result[i]['probabilities']),'\n')
      if(predict_result[i]['classes'] == labels[i]):
          counter+=1
  print('Accuracy: ',counter/len(predict_result))

if __name__ == "__main__":
  tf.app.run()
