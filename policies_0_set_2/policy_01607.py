def policy(env):
    # Strategy: Prioritize attacking when safe (<=1 adjacent monsters) to maximize damage and avoid unnecessary movement.
    # If surrounded (>=2 adjacent), move to the safest adjacent cell (fewest monsters) to minimize damage taken.
    # Always attack if no safe move exists, as dealing damage is better than doing nothing.
    if env.game_over:
        return [0, 0, 0]
    player_pos = env.player_pos
    monsters = env.monsters
    if not monsters:
        return [0, 0, 0]
    
    adjacent_count = 0
    for m in monsters:
        dx = abs(player_pos[0] - m['pos'][0])
        dy = abs(player_pos[1] - m['pos'][1])
        if dx + dy == 1:
            adjacent_count += 1
            
    if adjacent_count <= 1:
        return [0, 1, 0]
    
    best_move = None
    best_adjacent = float('inf')
    for move in [1, 2, 3, 4]:
        nx, ny = player_pos[0], player_pos[1]
        if move == 1: ny -= 1
        elif move == 2: ny += 1
        elif move == 3: nx -= 1
        elif move == 4: nx += 1
            
        if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
            continue
        occupied = any(nx == m['pos'][0] and ny == m['pos'][1] for m in monsters)
        if occupied:
            continue
            
        adj_count = 0
        for m in monsters:
            dx = abs(nx - m['pos'][0])
            dy = abs(ny - m['pos'][1])
            if dx + dy == 1:
                adj_count += 1
                
        if adj_count < best_adjacent:
            best_adjacent = adj_count
            best_move = move
            
    if best_move is not None:
        return [best_move, 0, 0]
    return [0, 1, 0]