import numpy as np
import random
import copy

def normal(x, mu, sig):
	return 1. / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * np.square(x - mu) / np.square(sig))


def trunc_normal(x, mu, sig, bounds=None):
	if bounds is None: 
		bounds = (-np.inf, np.inf)

	norm = normal(x, mu, sig)
	norm[x < bounds[0]] = 0
	norm[x > bounds[1]] = 0

	return norm


def sample_trunc(n, mu, sig, bounds=None):
	""" Sample `n` points from truncated normal distribution """
	x = np.linspace(mu - 5. * sig, mu + 5. * sig, 10000)
	y = trunc_normal(x, mu, sig, bounds)
	y_cum = np.cumsum(y) / y.sum()

	yrand = np.random.rand(n)
	sample = np.interp(yrand, y_cum, x)

	return sample


def activation_function(x):
	return 1/(1 + np.exp(-x))

class Nnetwork:
	"""
	A class to implement a neural network, with 1 hidden layer

	TODO (can try):
		Add epsilon for mutation, and epsilon's std deviation will be inversely proportional to its fitness function.
	"""
	def __init__(self, 
				 no_of_in_nodes, 
				 no_of_out_nodes, 
				 no_of_hidden_nodes):
		self.no_of_in_nodes = no_of_in_nodes
		self.no_of_out_nodes = no_of_out_nodes
		self.no_of_hidden_nodes = no_of_hidden_nodes
		self.create_weight_matrices()
		self.generation = 0
		self.score = 0
	
	def __lt__(self, other):
		return True
	
	def create_weight_matrices(self):
		""" A method to initialize the weight matrices of the neural network"""

		if True:
			# Best weights found
			self.weights_in_hidden, self.weights_hidden_out = ([[ 0.7184788 , -0.39752857, -0.47985157,  0.83834627,  0.39523845],
       [ 0.66627134,  0.78094477,  0.05071419, -0.54042169,  0.37841983],
       [ 0.03780728,  0.21773755, -0.28842893,  0.72384008,  0.45993991],
       [ 0.04301163,  0.93032421,  0.76279433, -0.88636449,  0.35971356],
       [ 0.53252678,  0.29918343,  0.23683955, -0.13565638, -0.45507588],
       [ 0.22397957,  0.13656037, -0.71989099,  0.117984  , -0.43427532],
       [-0.52184697, -0.78862114, -0.98496777,  0.30178235,  0.39502922],
       [ 0.09135914, -0.76757145, -0.09655984, -0.54047288,  0.68036377]],[[-0.04297519, -0.10843757,  0.76044326,  0.01445383, -0.40428726,
         0.28349921, -0.44160214, -0.92575841]])
			'''([[-0.13556458, -0.93175286, -0.39126295, -0.77406043, -0.43266956],
       [ 0.77724429,  0.8020252 , -0.52996272, -0.39265316,  0.18475643],
       [-0.21300316, -0.84447501,  0.83731892,  0.18156271,  0.30544719],
       [ 0.06102827, -0.7482265 ,  0.10946006,  0.91168808,  0.69756287],
       [ 0.05714248, -0.48387763,  0.84946712, -0.99462502,  0.35261359],
       [-0.64363127, -0.65122444, -0.89222938,  0.37215972, -0.16042259],
       [-0.05380377,  0.90047596, -0.51158714,  0.5909727 , -0.82038859],
       [ 0.80506825,  0.35550777, -0.12867547,  0.75514793,  0.23458062]],[[-0.86283071,  0.45088169, -0.11922221, -0.80138509, -0.22830238,
        -0.85155329,  0.90803623, -0.46110185]])'''
			
			self.weights = [self.weights_in_hidden, self.weights_hidden_out]
			return
		
		self.weights_in_hidden = np.zeros((self.no_of_hidden_nodes, self.no_of_in_nodes))
		self.weights_hidden_out = np.zeros((self.no_of_out_nodes, self.no_of_hidden_nodes))
		
		in_samples = sample_trunc(self.no_of_hidden_nodes * self.no_of_in_nodes, 0, 1, (-1, 1))
		k = 0
		for i in range(len(self.weights_in_hidden)):
			for j in range(len(self.weights_in_hidden[i])):
				self.weights_in_hidden[i][j] = in_samples[k]
				k += 1

		out_samples = sample_trunc(self.no_of_out_nodes * self.no_of_hidden_nodes, 0, 1, (-1, 1))
		k = 0
		for i in range(len(self.weights_hidden_out)):
			for j in range(len(self.weights_hidden_out[i])):
				self.weights_hidden_out[i][j] = out_samples[k]
				k += 1

		self.weights = [self.weights_in_hidden, self.weights_hidden_out]
	
	def get_matrices(self):
		return self.weights, self.score, self.generation
	
	def set_matrices(self, weights, generation):
		assert(self.generation == generation - 1)

		self.generation = generation

		self.weights = weights

	def set_scores(self, score):
		self.score = score

	def cross_breed(self, weights1, p):
		weights = []
		for i in range(len(weights1)):
			assert(weights1[i].shape == self.weights[i].shape)
			nums = np.random.choice([0, 1], size=weights1[i].shape, p=[1-p, p])

			weights.append(weights1[i] * nums + (1-nums) * self.weights[i])

		self.weights = weights
		return weights

	def run(self, inputs):
		assert(len(inputs) == self.no_of_in_nodes)
		input_vector = np.array(inputs, ndmin=2).T
		for wgt in self.weights:
			input_vector = activation_function(wgt @   input_vector)
			# output_vector = activation_function(self.weights_hidden_out @ input_hidden)
		return input_vector
