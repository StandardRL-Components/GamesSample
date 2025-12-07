def policy(env):
    # Strategy: Always move towards the nearest gem to maximize collection rate and minimize steps.
    # If on a gem, do nothing to collect it. Otherwise, choose the movement that minimizes Manhattan distance to the nearest gem.
    player_pos = env.player_pos
    gems = env.gems
    
    # Check if player is on a gem
    for gem in gems:
        if (player_pos == gem['pos']).all():
            return [0, 0, 0]
    
    # Find nearest gem
    min_dist = float('inf')
    nearest_gem = None
    for gem in gems:
        dist = abs(player_pos[0] - gem['pos'][0]) + abs(player_pos[1] - gem['pos'][1])
        if dist < min_dist:
            min_dist = dist
            nearest_gem = gem['pos']
    
    # Evaluate movement actions (excluding no-op)
    best_action = 0
    best_dist = float('inf')
    actions = [1, 2, 3, 4]  # up, down, left, right
    
    for a in actions:
        new_pos = player_pos.copy()
        if a == 1:  # up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif a == 2:  # down
            new_pos[1] = min(env.GRID_SIZE - 1, new_pos[1] + 1)
        elif a == 3:  # left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif a == 4:  # right
            new_pos[0] = min(env.GRID_SIZE - 1, new_pos[0] + 1)
        
        dist = abs(new_pos[0] - nearest_gem[0]) + abs(new_pos[1] - nearest_gem[1])
        if dist < best_dist:
            best_dist = dist
            best_action = a
    
    return [best_action, 0, 0]