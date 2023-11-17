MAX, MIN = 10000000, -10000000

def minimax(state, maximizingPlayer, 
            alpha, beta, get_neighbours, max_depth): 
    """runs minimax for adversarial games to get optimal value for the player at the root node

    Args:
        state (customizable struct): denotes a game(board) state
        maximizingPlayer (bool): is the node min or max 
        alpha (integer value): alpha for pruning
        beta (integer value): beta for pruning
        get_neighbours (function): function for generating neighbouring states
        max_depth (integer): maximum depth of the minimax tree

    Returns:
        integer value: value of the node (min/max)
    """
    # Terminating condition. i.e 
    # leaf node is reached 
    if state.depth == max_depth:
        return state.score
 
    if maximizingPlayer: 
        best = MIN

        for new_state in get_neighbours(state, maximizingPlayer):
            val = minimax(new_state,
                          False, alpha, beta, get_neighbours,
                          max_depth) 
            best = max(best, val) 
            alpha = max(alpha, best) 

            # Alpha Beta Pruning 
            if beta <= alpha: 
                break
          
        return best 
      
    else:
        best = MAX
        for new_state in get_neighbours(state, maximizingPlayer):
            val = minimax(new_state,
                          True, alpha, beta, get_neighbours,
                          max_depth) 
            best = min(best, val) 
            beta = min(beta, best) 

            # Alpha Beta Pruning 
            if beta <= alpha: 
                break
                           
        return best 
