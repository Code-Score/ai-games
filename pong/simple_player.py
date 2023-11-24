
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
        self.id = id

    def update_state(self, **kwargs):
        ballx = kwargs['ballx'] 
        bally = kwargs['bally']
        speedx = kwargs['speedx'] 
        speedy = kwargs['speedy'] 
        playerx = kwargs['playerx'] 
        playery = kwargs['playery']
        playerheight = kwargs['playerheight']

        player_centre = playery + (playerheight//2)
        if (bally > player_centre and speedy > 0):
            return 1
        elif (bally < player_centre and speedy < 0):
            return -1
        return 0

    def can_start(self):
        return True