#!/bin/python2

import struct
import glob
import os
import numpy as np
import random

class dataSet(object):
	def __init__(self):
		self.mods = os.listdir('data_radio')	#change to directory of training dataset
		self.exampleMap = {}
		self.examples = {}
		for mod in self.mods:
			#self.exampleMap[mod] = 'data_cat' + '/' + mod + '/' + 'ex.bin'
			self.exampleMap[mod] = 'data_radio' + '/' + mod + '/' + 'example_radio_xmit_1.bin'
			self.examples[mod] = self.readIEEEfile(self.exampleMap[mod])
			self.examples[mod] = self.examples[mod]/np.var(self.examples[mod])

				#change to directory of training dataset

	def readIEEEfile(self, fName):
		f = open(fName, 'rb')
		n = 600000			#number of bytes read in/2
		x = list(struct.unpack('f'*n, f.read(4*n)))
		x = np.atleast_3d(np.transpose(np.reshape(np.array(x), [n/2,2])))
		return x

	def getSubBatch(self, mod, nExamples):
		#print mod
		x = self.examples[mod]
		#print x.shape
		examples = []
		for i in xrange(0, nExamples):
			start = random.randint(0,220000 - 256)		#220000 #randomly choose a start index
			examples.append(x[:, start:start + 256, :])
			#print examples[-1].shape		#returns random slice of length 128
		return np.stack(examples)

	def getBatch(self, nExamplesPerBatch):
		subBatches = []
		labels = []
		for i, mod in enumerate(self.mods):
			subBatches.append(self.getSubBatch(mod, nExamplesPerBatch))
			labels.append(i*np.ones(nExamplesPerBatch))

		batch = np.concatenate(subBatches, axis=0)
		labels = np.concatenate(labels, axis=0)
		return batch, labels

class testSet(object):
	def __init__(self):
		self.mods = os.listdir('data_radio')	#change to directory of test dataset
		self.exampleMap = {}
		self.examples = {}
		for mod in self.mods:
			self.exampleMap[mod] = 'data_radio' + '/' + mod + '/' + 'example_radio_xmit_1.bin'
			self.examples[mod] = self.readIEEEfile(self.exampleMap[mod])
			self.examples[mod] = self.examples[mod]/np.var(self.examples[mod])

				#change to directory of test dataset

	def readIEEEfile(self, fName):
		f = open(fName, 'rb')
		n = 600000
		x = list(struct.unpack('f'*n, f.read(4*n)))
		x = np.atleast_3d(np.transpose(np.reshape(np.array(x), [n/2,2])))
		return x

	def getSubBatch(self, mod, pos, nExamples):
		x = self.examples[mod]
		#print x.shape
		examples = []
		start = 260000
		for i in xrange(0, nExamples*256, 256):
			examples.append(x[:, start + i:start + i + 256, :])
			#print examples[-1].shape		#returns random slice of length 128
		return np.stack(examples)

	def getBatch(self, pos, nExamplesPerBatchPerMod):
		subBatches = []
		labels = []
		for i, mod in enumerate(self.mods):
			subBatches.append(self.getSubBatch(mod, pos, nExamplesPerBatchPerMod))
			labels.append(i*np.ones(nExamplesPerBatchPerMod))

		batch = np.concatenate(subBatches, axis=0)
		labels = np.concatenate(labels, axis=0)
		return batch, labels

def main():
	data = dataSet()
	test = testSet()
	x, labels = data.getBatch(10)
	print x.shape
	print labels
	for pos in xrange(220000, 300000, 256*10):
	 	x, labels = test.getBatch(pos,10)	
	print x.shape
	print labels

if __name__ == "__main__":
	main()
