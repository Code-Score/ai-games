class Player:
    
    def __init__(self):
		"""
		Initilizes the variables needed.
		"""
        self.generation = 0
    
    def can_start(self):
		"""
		This function is called whenever a new game needs to start. 
        We can do any post/pre processing in this function.

        Returns:
            bool: whether we can start next game or not.
		"""
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
                 if  pipe['y'] <= y + 50:
                     return True
                 break
        
        return False # if not move
