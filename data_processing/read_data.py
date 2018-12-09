#!/bin/python2

import struct
import glob
import os
import numpy as np
import random

class dataSet(object):
	def __init__(self):
		self.mods = os.listdir('data_0dB_SNR')	#change to directory of training dataset
		self.exampleMap = {}
		for mod in self.mods:
			self.exampleMap[mod] = glob.glob('data_0dB_SNR' + '/' + mod + '/' + '*.bin')
				#change to directory of training dataset

	def readIEEEfile(self, fName):
		f = open(fName, 'rb')
		n = 256
		x = list(struct.unpack('f'*n, f.read(4*n)))
		x = np.atleast_3d(np.transpose(np.reshape(np.array(x), [n/2,2])))
		return x

	def getSubBatch(self, mod, nExamples):
		x = map(self.readIEEEfile, random.sample(self.exampleMap[mod], 2*nExamples))
		
		#print x
		x = np.stack(x)
		# shape is 2*nExamples, 2, 128, 1
		x = np.transpose(x, axes=[0, 2, 1, 3])
		x = np.reshape(x, (nExamples, 256, 2, 1))
		x = np.transpose(x, axes=[0, 2, 1, 3])
		#print x.shape

		start = random.randint(0,127)		#randomly choose a start index
		#start = 0 

		return x[:, :, start:start + 128, :]		#returns random slice of length 128

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
		self.mods = os.listdir('test_0dB_SNR')	#change to directory of test dataset
		self.exampleMap = {}
		for mod in self.mods:
			self.exampleMap[mod] = glob.glob('test_0dB_SNR' + '/' + mod + '/' + '*.bin')
				#change to directory of test dataset

	def readIEEEfile(self, fName):
		f = open(fName, 'rb')
		n = 256
		x = list(struct.unpack('f'*n, f.read(4*n)))
		x = np.atleast_3d(np.transpose(np.reshape(np.array(x), [n/2,2])))
		return x

	def getSubBatch(self, mod, pos, nExamples):
		#for loop over the examples
			x = map(self.readIEEEfile, self.exampleMap[mod][pos:pos+2*nExamples])
			#print x
			x = np.stack(x)
			# shape is 2*nExamples, 2, 128, 1
			x = np.transpose(x, axes=[0, 2, 1, 3])
			x = np.reshape(x, (nExamples, 256, 2, 1))
			x = np.transpose(x, axes=[0, 2, 1, 3])
			#print x.shape

			start = random.randint(0,127)
		#randomly choose a start index (same pattern for each test)
		#start = 0 

			return x[:, :, start:start + 128, :]		#returns random slice of length 128

	def getBatch(self, pos, nExamplesPerBatch):
		subBatches = []
		labels = []
		for i, mod in enumerate(self.mods):
			subBatches.append(self.getSubBatch(mod, pos, nExamplesPerBatch))
			labels.append(i*np.ones(nExamplesPerBatch))

		batch = np.concatenate(subBatches, axis=0)
		labels = np.concatenate(labels, axis=0)
		return batch, labels

def main():
	data = dataSet()
	test = testSet()
	#x, labels = data.getBatch(10)
	for pos in xrange(0, len(test.exampleMap[test.mods[0]]), 2*10):
		x, labels = test.getBatch(pos,10)	
	print x.shape

if __name__ == "__main__":
	main()