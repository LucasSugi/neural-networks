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

filter_layer_1 = 64
filter_layer_2 = 128
filter_layer_3 = 256
units_dense_1 = 512
units_dense_2 = 1024
epochs = 1

#Show log's on windows
tf.logging.set_verbosity(tf.logging.INFO)

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
  # Load training and eval data
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  #Change dtype
  x_train = np.float32(x_train)
  x_test = np.float32(x_test)
  y_train = np.int32(y_train)
  y_test = np.int32(y_test)

  #Normalize
  x_train = normalize(x_train)
  x_test = normalize(x_test)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='model_cifar10',warm_start_from='model_cifar10')

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_train,y=y_train,num_epochs=epochs,shuffle=True)
  cifar10_classifier.train(input_fn=train_input_fn,hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_test,y=y_test,num_epochs=1,shuffle=False)
  eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
