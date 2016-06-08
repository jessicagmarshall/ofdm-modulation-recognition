#!/usr/bin/python2

import tensorflow as tf
import numpy as np
import random 
import matplotlib.pyplot as plt

from ops import *
from read_data import dataSet, testSet
from sklearn.metrics import confusion_matrix 
# def getExample(batch_size):
# 	batch = np.random.randn(batch_size, 2, 128, 1)
# 	labels = np.random.random_integers(0,10,size=(batch_size))
# 	batch = batch + 10*labels[:, np.newaxis, np.newaxis, np.newaxis]
# 	return batch, labels


class modulation_net(object):
	def __init__(self, sess, nSamples, nClasses, examples_per_class):
		self.sess = sess
		self.nSamples = nSamples
		self.nClasses = nClasses
		self.data = dataSet()
		self.testset = testSet()
		self.examples_per_class = examples_per_class
		self.train_batch_size = self.data.getBatch(examples_per_class)[0].shape[0]

		self.msg = tf.placeholder(tf.float32, shape=(self.train_batch_size,2,self.nSamples,1))		
		self.y = tf.placeholder(tf.int64, shape=(self.train_batch_size))

		self.train_net = self.build_net(self.train_batch_size)

		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.train_net, self.y))
		self.optim = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.loss)
		# self.counter = 1
		tf.initialize_all_variables().run()


	def build_net(self, batch_size, train=True):
		if not train:
			tf.get_variable_scope().reuse_variables()

		conv1 = tf.nn.elu(conv2d(self.msg, 64, k_h=1, k_w=3, d_h=1, d_w=1, name='conv1'))

		conv1_padded = tf.pad(conv1, [[0, 0], [0, 0], [1, 1] , [0, 0]])
		conv2 = tf.nn.elu(conv2d(conv1_padded, 16, k_h=2, k_w=3, d_h=1, d_w=1, padding='VALID', name='conv2'))

		conv2_flat = tf.reshape(conv2,[batch_size,-1])

		dense1 = tf.nn.elu(linear(conv2_flat, 128, name='lin1'))
		output = linear(dense1, self.nClasses, name='lin2')

		print self.msg.get_shape()
		print conv1.get_shape()
		print conv1_padded.get_shape()
		print conv2.get_shape()
		print conv2_flat.get_shape()
		print dense1.get_shape()
		print output.get_shape()

		return output


	def train_iter(self, example_batch, label_batch):
		loss, _ = self.sess.run([self.loss, self.optim], feed_dict={self.msg : example_batch, self.y : label_batch})		
		print loss

	def train(self, nBatches):
		for i in xrange(0,nBatches):
			# batch, labels = getExample(self.train_batch_size)
			batch, labels = self.data.getBatch(self.examples_per_class)
			model.train_iter(batch, labels)

	def test(self):
		random.seed(31415)
		self.test_net = self.build_net(self.train_batch_size, train=False)
		self.test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.test_net, self.y))
		dec = tf.argmax(tf.nn.softmax(self.test_net),1)
		#batch, labels = getExample(self.train_batch_size)
		#batch, labels = self.test.getBatch(self.examples_per_class)
		testloss = []
		cm = np.zeros((6,6))
		print 'testing'
		for pos in xrange(0, len(self.testset.exampleMap[self.testset.mods[0]]), 2*self.examples_per_class):
			#print pos
			batch, labels = self.testset.getBatch(pos,self.examples_per_class)
			loss, dec_batch = self.sess.run([self.test_loss, dec], feed_dict={self.msg : batch, self.y : labels})		
			cm_batch = confusion_matrix(labels, dec_batch)
			cm = cm + cm_batch
			testloss.append(loss)
		print 'confusion matrix\n', cm/np.sum(cm)*np.shape(cm)[0]
		print 'percent correct ', 100*np.sum(np.diagonal(cm)) / np.sum(cm), '%\n'
		print 'test error', sum(testloss)/len(testloss)
		plt.figure()
		plot_confusion_matrix(cm*nClasses/np.sum(cm), title='Confusion Matrix (0dB SNR)')
		plt.show()

def plot_confusion_matrix(cm, title='Confusion Matrix (0dB SNR)', cmap=plt.cm.Blues):
    	plt.imshow(cm, interpolation='nearest', cmap=cmap)
    	plt.title(title)
    	plt.colorbar()
    	tick_marks = np.arange(0, 6)
    	plt.xticks(tick_marks, ('BPSK', 'QPSK', '16QAM', '64QAM', '128QAM', '256QAM'), rotation=45)
    	plt.yticks(tick_marks,  ('BPSK', 'QPSK', '16QAM', '64QAM', '128QAM', '256QAM'))
    	plt.tight_layout()
    	plt.ylabel('True label')
    	plt.xlabel('Predicted label')

nClasses = 6 # number modulation schemes to detect
nSamples = 128 # number of samples per example

# train_net = build_net(nSamples, nClasses, 1024)
# test_net = build_net(nSamples, nClasses, 32, train=False)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
	model = modulation_net(sess, nSamples, nClasses, 10)
	model.train(1000)
	model.test()