def policy(env):
    # Strategy: Move towards nearest gem and collect when on top. Prioritize immediate collection for +1 reward,
    # then move to reduce Euclidean distance for +0.1 shaping reward. Avoid walls and use grid boundaries.
    a1 = 1 if any(gem['pos'] == env.cursor_pos for gem in env.gems) else 0
    if a1 == 1:
        return [0, a1, 0]  # Collect gem, no movement needed
    
    if not env.gems:
        return [0, 0, 0]  # No gems, wait
    
    # Find nearest gem by Euclidean distance
    cursor = env.cursor_pos
    nearest_gem = min(env.gems, key=lambda g: (g['pos'][0]-cursor[0])**2 + (g['pos'][1]-cursor[1])**2)
    target = nearest_gem['pos']
    
    # Evaluate movement actions (excluding no-op)
    best_action = 0
    best_dist = float('inf')
    for action in [1, 2, 3, 4]:
        new_pos = list(cursor)
        if action == 1: new_pos[1] = max(0, new_pos[1]-1)
        elif action == 2: new_pos[1] = min(env.GRID_SIZE[1]-1, new_pos[1]+1)
        elif action == 3: new_pos[0] = max(0, new_pos[0]-1)
        elif action == 4: new_pos[0] = min(env.GRID_SIZE[0]-1, new_pos[0]+1)
        
        # Skip if movement is blocked (wall)
        if new_pos == cursor:
            continue
            
        dist = (target[0]-new_pos[0])**2 + (target[1]-new_pos[1])**2
        if dist < best_dist:
            best_dist = dist
            best_action = action
            
    return [best_action, 0, 0]