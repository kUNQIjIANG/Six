from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the weights and biases
layer_1_weights = tf.Variable(np.loadtxt('hidden_layer_1_weights.txt', dtype='float32'))
layer_1_biases = tf.Variable(np.loadtxt('hidden_layer_1_biases.txt', dtype='float32'))
layer_2_weights = tf.Variable(np.loadtxt('hidden_layer_2_weights.txt', dtype='float32'))
layer_2_biases = tf.Variable(np.loadtxt('hidden_layer_2_biases.txt', dtype='float32'))
output_layer_weights = tf.Variable(np.loadtxt('output_layer_weights.txt', dtype='float32'))
output_layer_biases = tf.Variable(np.loadtxt('output_layer_biases.txt', dtype='float32'))

# load the mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

INPUT_SIZE = mnist.test.images.shape[1]
OUTPUT_SIZE = mnist.test.labels.shape[1]

# create placeholders for inputs to the TensorFlow graph
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

# Layer 1
layer_1_out = tf.nn.sigmoid(tf.matmul(x,layer_1_weights) + layer_1_biases)

# Layer 2
layer_2_out = tf.nn.relu(tf.matmul(layer_1_out, layer_2_weights) + layer_2_biases)

# output layer
soft_out = tf.nn.softmax(tf.matmul(layer_2_out, output_layer_weights) + output_layer_biases)
out = tf.matmul(layer_2_out, output_layer_weights) + output_layer_biases

# cross-entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = out))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# example of a simple model accuracy test,
# -- assumes the output tensor from the model above is called `out`
with tf.name_scope('results'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(soft_out, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

# run the model
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		batch = mnist.train.next_batch(100)
		if i % 20 == 0:
			train_accuracy = accuracy.eval(feed_dict = {x : batch[0], y: batch[1]})
			print("step %d, train_accuracy: %g" %(i, train_accuracy))
		train_step.run(feed_dict = {x:batch[0], y:batch[1]})
	print('accuracy:{}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))

# result : 95.45%
