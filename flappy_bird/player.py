# Import python libraries required in this example:
import time
import signal
import sys
from neural_network import Nnetwork

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
							no_of_hidden_nodes=8) 
					for i in range(self.num_players)
				  ]
		self.current_player = -1
		self.best_score = None
		self.best_player = self.current_player
		self.start_time = time.time()
		self.generation += 1

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
			if (self.best_score is None or ( self.best_score < end_time - self.start_time)):
				self.best_score = end_time - self.start_time
				self.best_player = self.current_player
			self.start_time = time.time()

		self.current_player += 1

		"""
		when all the players in a generation finish playing, assign this generation's best player as next generation's first player
		so that we are always improving our best player across generations
		"""
		if self.current_player >= self.num_players:
			best_player = self.players[self.best_player]
			
			self.reset()
			
			self.current_player += 1
			self.players[0] = best_player
			
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
