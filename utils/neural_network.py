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
				 no_of_hidden_nodes,
				 best_weights = None):
		self.no_of_in_nodes = no_of_in_nodes
		self.no_of_out_nodes = no_of_out_nodes
		self.no_of_hidden_nodes = no_of_hidden_nodes
		if (best_weights is not None):
			self.weights = best_weights
		else:
			self.create_weight_matrices()
		self.generation = 0
		self.score = 0
	
	def __lt__(self, other):
		return True
	
	def create_weight_matrices(self):
		""" A method to initialize the weight matrices of the neural network"""

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
