def policy(env):
    # Optimized policy: Prioritize immediate gem collection while avoiding traps. 
    # Uses Manhattan distance to evaluate moves, with heavy penalty for traps and bonus for gems.
    # Balances exploration (moving towards closest gem) with safety (maintaining distance from traps).
    if env.game_over:
        return [0, 0, 0]
    
    current_pos = env.player_pos
    gems = [gem['pos'] for gem in env.gem_pos]
    traps = env.trap_pos
    
    # Calculate current distances
    current_gem_dist = min([abs(current_pos[0]-g[0]) + abs(current_pos[1]-g[1]) for g in gems]) if gems else float('inf')
    current_trap_dist = min([abs(current_pos[0]-t[0]) + abs(current_pos[1]-t[1]) for t in traps]) if traps else float('inf')
    
    best_action = 0
    best_score = -float('inf')
    
    moves = [0, 1, 2, 3, 4]
    for move in moves:
        if move == 0:
            new_pos = current_pos
        elif move == 1:
            new_pos = (current_pos[0], current_pos[1]-1)
        elif move == 2:
            new_pos = (current_pos[0], current_pos[1]+1)
        elif move == 3:
            new_pos = (current_pos[0]-1, current_pos[1])
        elif move == 4:
            new_pos = (current_pos[0]+1, current_pos[1])
            
        # Check if move is valid (within bounds and not a wall)
        if (new_pos[0] < 0 or new_pos[0] >= env.MAZE_WIDTH or 
            new_pos[1] < 0 or new_pos[1] >= env.MAZE_HEIGHT or 
            env.maze[new_pos[1]][new_pos[0]] == 1):
            continue
            
        new_gem_dist = min([abs(new_pos[0]-g[0]) + abs(new_pos[1]-g[1]) for g in gems]) if gems else float('inf')
        new_trap_dist = min([abs(new_pos[0]-t[0]) + abs(new_pos[1]-t[1]) for t in traps]) if traps else float('inf')
        
        # Calculate score with emphasis on safety and gem collection
        dist_reward = (current_gem_dist - new_gem_dist) * 2.0  # Prioritize gem proximity
        safety_penalty = 0.5 * (current_trap_dist - new_trap_dist)  # Penalize moving closer to traps
        
        gem_bonus = 15 if new_pos in gems else 0  # Higher bonus for immediate collection
        trap_penalty = -1000 if new_pos in traps else 0  # Very heavy penalty for traps
        
        score = dist_reward + safety_penalty + gem_bonus + trap_penalty
        
        if score > best_score:
            best_score = score
            best_action = move
            
    return [best_action, 0, 0]