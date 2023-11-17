from rlrisk.agents import *
from rlrisk.environment import *
from player import Player

players = [Player(), Player()] # [BaseAgent(), AggressiveAgent(), Human()]
env = Risk(players, has_gui=True)
results = env.play()
