def policy(env):
    # Greedy policy: Move towards nearest crystal while avoiding traps. Prioritizes crystal collection
    # by minimizing Manhattan distance, with trap avoidance. Breaks ties by movement priority.
    player_pos = env.player_pos
    crystals = env.crystals
    traps = env.traps
    
    if not crystals:
        return [0, 0, 0]
    
    # Calculate distances to all crystals and traps
    crystal_dists = [abs(player_pos[0]-c[0]) + abs(player_pos[1]-c[1]) for c in crystals]
    trap_dists = [abs(player_pos[0]-t[0]) + abs(player_pos[1]-t[1]) for t in traps]
    
    # Find nearest crystal and its direction
    nearest_crystal = crystals[np.argmin(crystal_dists)]
    dx = nearest_crystal[0] - player_pos[0]
    dy = nearest_crystal[1] - player_pos[1]
    
    # Generate candidate moves (0: none, 1: up, 2: down, 3: left, 4: right)
    moves = []
    if dx < 0: moves.append(1)  # up
    if dx > 0: moves.append(2)  # down
    if dy < 0: moves.append(3)  # left
    if dy > 0: moves.append(4)  # right
    if not moves: moves = [0]  # no movement needed
    
    # Evaluate moves: avoid traps, prefer crystal direction
    best_move = 0
    best_score = -float('inf')
    for move in moves:
        new_x, new_y = player_pos
        if move == 1: new_x -= 1
        elif move == 2: new_x += 1
        elif move == 3: new_y -= 1
        elif move == 4: new_y += 1
        
        # Check bounds
        new_x = max(0, min(env.grid_size[0]-1, new_x))
        new_y = max(0, min(env.grid_size[1]-1, new_y))
        new_pos = (new_x, new_y)
        
        # Skip if trap
        if new_pos in traps:
            continue
            
        # Score: negative distance to crystal + small tie-breaker
        dist = abs(new_x - nearest_crystal[0]) + abs(new_y - nearest_crystal[1])
        score = -dist - 0.1 * move  # tie-breaker: prefer lower move index
        if score > best_score:
            best_score = score
            best_move = move
    
    # Fallback if all moves are traps
    if best_score == -float('inf'):
        return [0, 0, 0]
        
    return [best_move, 0, 0]