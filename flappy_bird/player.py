# Import python libraries required in this example:
import time
import signal
import sys
sys.path.append('..')
from utils.neural_network import Nnetwork
import heapq
from copy import deepcopy

BEST_WEIGHTS = [
				[[-0.57294702,  0.5335062 , -0.26320206,  0.82531126, -0.52839151],
				[-0.23820509, -0.39910418,  0.6001955 , -0.12010982,  0.2451838 ],
				[-0.06752973, -0.67686273,  0.55676117, -0.50734295, -0.64441767],
				[-0.42010694,  0.15328949, -0.19522216,  0.78951863,  0.80408268],
				[-0.37420325, -0.45379166, -0.08607935, -0.94911611,  0.50238367],
				[-0.61256153,  0.47756111,  0.33115688, -0.4781571 , -0.80569627],
				[ 0.58315541, -0.82892777,  0.48342056, -0.36501762,  0.27305253],
				[-0.14841321,  0.77485721, -0.23566494, -0.74901839, -0.16059864]],
				[[-0.01179975,  0.06162142,  0.24259025,  0.83968532, -0.44324076,
						0.97067869, -0.39917555, -0.93976672]]
			]

class Player:
	
	def __init__(self):
		"""
		Initilizes the variables needed.
		"""
		self.generation = 0
		self.reset()
		signal.signal(signal.SIGINT, lambda x, y: self.signal_handler(x, y))
	
	def signal_handler(self, sig, frame):
		print('You pressed Ctrl+C!')
		print(self.players[self.current_player].get_matrices())
		sys.exit(0)


	def reset(self):
		"""
		Resets all the variables for a new game.
		"""
		self.num_players = 100
		self.players = [Nnetwork(no_of_in_nodes=5, 
							no_of_out_nodes=1, 
							no_of_hidden_nodes=8, best_weights = None) # set best_weights = BEST_WEIGHTS to run pre-trained player
					for i in range(self.num_players)
				  ]
		self.current_player = -1
		self.best_score = None
		self.best_player = self.current_player
		self.start_time = time.time()
		self.generation += 1
		self.sorted_players = []

	def can_start(self):
		"""
		This function is called whenever a new game needs to start. 
		We can do any post/pre processing in this function.

		Returns:
			bool: whether we can start next game or not.
		"""

		# keep track of the best player of this generation (have used time as the fitness metric)
		if self.current_player == -1:
			self.start_time = time.time()
		else:
			end_time = time.time()
			score = end_time - self.start_time
			heapq.heappush(self.sorted_players, (-score, self.players[self.current_player]))

			if (self.best_score is None or ( self.best_score < end_time - self.start_time)):
				self.best_score = score
				self.best_player = self.current_player
			self.start_time = time.time()

		self.current_player += 1

		"""
		when all the players in a generation finish playing, assign this generation's best player as next generation's first player
		so that we are always improving our best player across generations
		"""
		if self.current_player >= self.num_players:
			best_player = self.players[self.best_player]

			assert(best_player == heapq.heappop(self.sorted_players)[1])
			
			second_best_player = heapq.heappop(self.sorted_players)[1]

			self.reset()
			
			self.current_player += 1
			self.players[0] = deepcopy(best_player)
			second_best_player.cross_breed(best_player.get_matrices()[0], 0.7)
			self.players[1] = deepcopy(second_best_player)
			
			for i in range(2, self.num_players - 20):
				self.players[i].cross_breed(best_player.get_matrices()[0], 0.9)
			
		return True
		
	def update_state(self, **kwargs):
		"""
		This function is called by the game to retrieve the next move.
		
		Args:
			up_pipes (list)  : List of all up pipes present on the screen. 
							   These can be ahead or behind the bird.
							   (x, y) contains the coordinates of bottom left point of the pipe.
			down_pipes (list): List of all down pipes present on the screen. 
							   These can be ahead or behind the bird.
							   (x, y) contains the coordinates of top left point of the pipe.
			x(int)			 : X position of the bird's leftmost point. 
			y(int)			 : Y position of the bird's uppermost point. 
			bird_height(int) : Bird's height.
			bird_width(int)  : Bird's width.
			pipe_height(int) : Pipe's height.
			pipe_width(int)  : Pipe's width.

		Returns:
			bool: Whether to move up or not.
		"""
		up_pipes = kwargs['up_pipes'] 
		down_pipes = kwargs['down_pipes'] 
		x = kwargs['x'] 
		y = kwargs['y']
		bird_height = kwargs['bird_height'] 
		bird_width = kwargs['bird_width'] 
		pipe_height = kwargs['pipe_height'] 
		pipe_width = kwargs['pipe_width']
		
		for pipe in down_pipes:
			if ((pipe['x'] >= x or pipe['x'] + pipe_width >= x)):
				if self.players[self.current_player].run(
					[pipe['x'],
					pipe['y'],
					x,
					y,
					pipe_width]
				)[0][0] > 0.5:
					return True
				break

		return False # if not move
