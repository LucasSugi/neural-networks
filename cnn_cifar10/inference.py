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

filter_layer_1 = 64
filter_layer_2 = 128
filter_layer_3 = 256
units_dense_1 = 512
units_dense_2 = 1024

#Model for cnn
def cnn_model_fn(features, labels, mode):
  # Convolutional Layer 1
  conv1 = tf.layers.conv2d(inputs=features,filters=filter_layer_1,kernel_size=[3,3],padding="same",activation=tf.nn.relu)

  # Pooling Layer 1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer 2
  conv2 = tf.layers.conv2d(inputs=pool1,filters=filter_layer_2,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Pooling Layer 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer 3
  conv3 = tf.layers.conv2d(inputs=pool2,filters=filter_layer_3,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Pooling Layer 2
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * filter_layer_3])

  # Dense Layer 1
  dense_1 = tf.layers.dense(inputs=pool3_flat, units=units_dense_1, activation=tf.nn.relu)
  
  # Dense Layer 2
  dense_2 = tf.layers.dense(inputs=dense_1, units=units_dense_2, activation=tf.nn.relu)

  # Add dropout operation; 0.80 probability that element will be kept
  dropout = tf.layers.dropout(inputs=dense_2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def normalize(data):
  #Normalize data
  for i in range(data.shape[0]):
      data[i,:,:,0] = (data[i,:,:,0]-np.min(data[i,:,:,0])) / (np.max(data[i,:,:,0])-np.min(data[i,:,:,0]))
      data[i,:,:,1] = (data[i,:,:,1]-np.min(data[i,:,:,1])) / (np.max(data[i,:,:,1])-np.min(data[i,:,:,1]))
      data[i,:,:,2] = (data[i,:,:,2]-np.min(data[i,:,:,2])) / (np.max(data[i,:,:,2])-np.min(data[i,:,:,2]))
  return data

def main(unused_argv):
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
  labels = np.asarray([0,1,2,3,4,5,6,7,8,9],dtype=np.int32)

  #Normalize
  data = normalize(data)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='model_cifar10',warm_start_from='model_cifar10')

  #Dictionary of classes
  keys = {0:'Airplane',1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'Dog',6:'Frog',7:'Horse',8:'Ship',9:'Truck'}

  # Predict
  predict_model = tf.estimator.inputs.numpy_input_fn(x=data,y=labels,num_epochs=1,shuffle=False)
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
