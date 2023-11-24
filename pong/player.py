
# Import python libraries required in this example:
import time
import signal
import sys
sys.path.append('..')
from utils.neural_network import Nnetwork
import heapq
from copy import deepcopy
import random

class Player:
        # Take the initial position, dimensions, speed and color of the object
    def __init__(self, id):
        """
        Initilizes the variables needed.
        """
        self.generation = 0
        self.reset()
        signal.signal(signal.SIGINT, lambda x, y: self.signal_handler(x, y))
        self.id = id
    
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        print(self.players[self.best_player].get_matrices())
        sys.exit(0)
    
    def reset(self):
        """
        Resets all the variables for a new game.
        """
        self.num_players = 1000
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
        self.sorted_players = []

    def update_state(self, **kwargs):
        """
        This function is called by the game to retrieve the next move.
        
        Args:
            ballx(int)             : X position of the ball 
            bally(int)             : Y position of the ball
            speedx(int)            : X component of ball's velocity
            speedy(int)            : Y component of ball's velocity
            playerx(int)           : X position of the player(leftmost point)
            playery(int)           : Y position of the player(topmost point)

        Returns:
            integer: to move up or down or don't move.
        """
        ballx = kwargs['ballx'] 
        bally = kwargs['bally']
        speedx = kwargs['speedx'] 
        speedy = kwargs['speedy'] 
        playerx = kwargs['playerx'] 
        playery = kwargs['playery']
        playerheight = kwargs['playerheight']
        
        q_val = self.players[self.current_player].run(
                                                    [ballx,
                                                    bally,
                                                    speedx,
                                                    speedy,
                                                    playery]
                                                )[0][0]
        if q_val > 0.6:
            return 1
        elif q_val < 0.4:
            return -1
        else:
            return 0

        return random.choice([-1,0,1])

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

            if (self.best_score is None or ( self.best_score < score)):
                self.best_score = score
                self.best_player = self.current_player
            self.start_time = time.time()

        self.current_player += 1

        """
        insert your logic to create players in next generation, based on the scores in last generation
        """
        if self.current_player >= self.num_players:
            bestplayer = self.players[self.best_player]
            print("best_score -> ", self.best_player, " , ", self.best_score)
            best_players = []
            n_best_players = self.num_players//10
            n_new_random_players = self.num_players//10
            for i in range(n_best_players):
                best_players.append(deepcopy(heapq.heappop(self.sorted_players)[1]))

            self.reset()
            
            self.current_player += 1
            for i in range(n_best_players):
                self.players[i] = deepcopy(best_players[i])

            for i in range(n_best_players, 2*n_best_players):
                self.players[i] = deepcopy(best_players[i%n_best_players])
                self.players[i].cross_breed(bestplayer.get_matrices()[0], 0.5)

            for i in range(2*n_best_players, self.num_players - n_new_random_players):
                self.players[i].cross_breed(best_players[i%n_best_players].get_matrices()[0], 0.9)
            
        return True