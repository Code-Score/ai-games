import random
import numpy as np
from copy import deepcopy
from logic import *
from minimax import minimax, MIN, MAX

moves = ["w", "a", "s", "d"]

Optimal_weights  = [[0.135, 0.121, 0.102, 0.0999],
                    [0.0997, 0.088, 0.076, 0.0724],
                    [0.0606, 0.0562, 0.0371, 0.0161],
                    [0.0125, 0.0099, 0.0057, 0.0033]]


class BoardState:
    def __init__(self, score, board, depth = 0):
        self.score = score
        self.board = board
        self.depth = depth

    def __gt__(self, o2):
        return self.score > o2.score

    def __lt__(self, o2):
        return self.score < o2.score
        

class Player:
    
    def __init__(self):
        """
        Initilizes the variables needed.
        """
        self.score = []
        self.max_depth = 7

    def can_start(self):
        """
        This function is called whenever a new game needs to start. 
        We can do any post/pre processing in this function.

        Returns:
            bool: whether we can start next game or not.
        """
        return True
    
    def update_score(self, score):
        """
        update_score is called after a game ends to update the scores.

        Args:
            score (int): score of the current game.
        """
        self.score.append(score)
        print(self.score)
    
    def get_score(self, board):
        return self.evalgrid(board)
        """
        # could be used as an alternative score function for evaluating a board
        num_empty = 0
        total_val = 0
        max_val = 0
        for i in range(4):
            for j in range(4):
                if (board[i][j] == 0):
                    num_empty += 1
                else:
                    total_val += board[i][j]
                    max_val = max(max_val, board[i][j])
        avg_val = (total_val)/(16-num_empty)

        possible_moves = 0
        for each in moves:
            new_board = move(each, deepcopy(board))
            if new_board != board:
                possible_moves += 1

        # weights can be tuned heuristically(grid search)
        return (20*possible_moves + 11*num_empty + 6*avg_val + 20*max_val)
        """

    def get_neighbours(self, state, maximizingPlayer):
        if maximizingPlayer:
            for each in moves:
                new_board = move(each, deepcopy(state.board))
                if new_board != state.board:
                    status = checkGameStatus(new_board)
                    if status not in ["PLAY", "WIN"]:
                        continue
                    new_state = BoardState(self.get_score(new_board), new_board, state.depth+1)

                    yield new_state
        else:
            for i in [2, 4]:
                for a in range(4):
                    for b in range(4):
                        if state.board[a][b] != 0:
                            continue
                        
                        new_board = deepcopy(state.board)
                        new_board[a][b] = i
                        
                        new_state = BoardState(self.get_score(new_board), new_board, state.depth+1)
                        
                        yield new_state
        return

    def evalgrid(self, grid):
        return np.sum(np.array(grid) * Optimal_weights)

    def update_state(self, **kwargs):
        """
        This function is called by the game to retrieve the next move.
        This also gets the result of the previous move.
        
        Args:
            board (list): contains the current state of the board.
        Returns:
            string: the next move ('w', 'a', 's', 'd').
        """
        board_ = kwargs["board"]

        state_ = BoardState(self.get_score(board_), board_)
        
        alpha = MIN
        beta = MAX
        best = MIN
        best_move = ''
        for each in moves:
            new_board = move(each, deepcopy(state_.board))
            if new_board != state_.board:
                status = checkGameStatus(new_board)
                if status not in ["PLAY", "WIN"]:
                    continue
                new_state = BoardState(self.get_score(new_board), new_board, state_.depth+1)
                val = minimax(new_state, 
                          False, alpha, beta, self.get_neighbours, self.max_depth) 
                if best < val:
                    best_move = each
                best = max(best, val) 
      
        if best_move == '':
            return random.choice(moves)
        else:
            return best_move