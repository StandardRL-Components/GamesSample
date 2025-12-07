def policy(env):
    """
    Strategy: Maximize reward by clearing largest possible groups. 
    For each step, evaluate current and adjacent cells' group sizes. 
    Move towards the largest group if it's larger than current, else clear current if viable (size>=2). 
    Avoid shift (a2=0) and only use space (a1=1) when beneficial.
    """
    r, c = env.selector_pos
    current_size = len(env._find_connected(r, c)) if env.grid[r, c] != 0 else 0
    
    best_size = current_size
    best_move = 0
    for move, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)], start=1):
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.GRID_ROWS and 0 <= nc < env.GRID_COLS:
            group_size = len(env._find_connected(nr, nc)) if env.grid[nr, nc] != 0 else 0
            if group_size > best_size:
                best_size = group_size
                best_move = move
    
    if best_size > current_size:
        return [best_move, 0, 0]
    elif current_size >= 2:
        return [0, 1, 0]
    else:
        for move in [4, 2, 3, 1]:
            nr, nc = r + (0 if move in [3,4] else (1 if move==2 else -1)), c + (0 if move in [1,2] else (1 if move==4 else -1))
            if 0 <= nr < env.GRID_ROWS and 0 <= nc < env.GRID_COLS:
                return [move, 0, 0]
        return [0, 0, 0]