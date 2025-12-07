def policy(env):
    # This policy uses one-step lookahead to maximize immediate correct pixels after a move.
    # It simulates each possible move (up/down/left/right) and selects the one yielding the highest match with the target.
    # If already solved, returns no-op to avoid wasting moves. Secondary actions are set to 1 (ignored by env).
    n = env.GRID_SIZE
    current = env.current_grid
    target = env.target_grid
    
    if (current == target).all():
        return [0, 1, 1]
    
    best_move = 0
    best_score = -1
    
    for move in [1, 2, 3, 4]:
        if move == 1:
            simulated = current[[(i+1) % n for i in range(n)], :]
        elif move == 2:
            simulated = current[[(i-1) % n for i in range(n)], :]
        elif move == 3:
            simulated = current[:, [(j+1) % n for j in range(n)]]
        else:
            simulated = current[:, [(j-1) % n for j in range(n)]]
        score = (simulated == target).sum()
        if score > best_score:
            best_score = score
            best_move = move
            
    return [best_move, 1, 1]