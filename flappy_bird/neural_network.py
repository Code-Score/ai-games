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
		
	def create_weight_matrices(self):
		""" A method to initialize the weight matrices of the neural network"""

		if False:
			# Best weights found
			self.weights_in_hidden, self.weights_hidden_out = ([[-0.57294702,  0.5335062 , -0.26320206,  0.82531126, -0.52839151],
		[-0.23820509, -0.39910418,  0.6001955 , -0.12010982,  0.2451838 ],
		[-0.06752973, -0.67686273,  0.55676117, -0.50734295, -0.64441767],
		[-0.42010694,  0.15328949, -0.19522216,  0.78951863,  0.80408268],
		[-0.37420325, -0.45379166, -0.08607935, -0.94911611,  0.50238367],
		[-0.61256153,  0.47756111,  0.33115688, -0.4781571 , -0.80569627],
		[ 0.58315541, -0.82892777,  0.48342056, -0.36501762,  0.27305253],
		[-0.14841321,  0.77485721, -0.23566494, -0.74901839, -0.16059864]], 
		[[-0.01179975,  0.06162142,  0.24259025,  0.83968532, -0.44324076,
			0.97067869, -0.39917555, -0.93976672]])
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

	def get_matrices(self):
		return self.weights_in_hidden, self.weights_hidden_out, self.score, self.generation
	
	def set_matrices(self, weights_in_hidden, weights_hidden_out, generation):
		assert(self.generation == generation - 1)

		self.generation = generation

		self.weights_in_hidden = copy.deepcopy(weights_in_hidden)
		self.weights_hidden_out = copy.deepcopy(weights_hidden_out)

	def set_scores(self, score):
		self.score = score

	def run(self, inputs):
		assert(len(inputs) == self.no_of_in_nodes)
		input_vector = np.array(inputs, ndmin=2).T
		input_hidden = activation_function(self.weights_in_hidden @   input_vector)
		output_vector = activation_function(self.weights_hidden_out @ input_hidden)
		return output_vector
