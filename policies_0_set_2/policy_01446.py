def policy(env):
    # Strategy: Move towards nearest gem while avoiding enemies. Prioritize immediate gem collection to maximize score multiplier.
    # Use Manhattan distance for efficiency. Break ties by preferring movement actions over no-op to avoid stagnation.
    if not env.gems:
        return [0, 0, 0]
    
    # Predict enemy positions for next step
    dangerous_positions = set()
    for enemy in env.enemies:
        if (env.steps + 1) % enemy['speed'] == 0:
            next_idx = (enemy['path_idx'] + 1) % len(enemy['path'])
            dangerous_positions.add(tuple(enemy['path'][next_idx]))
        else:
            dangerous_positions.add(tuple(enemy['pos']))
    
    current_pos = env.player_pos
    gems_pos = [gem['pos'] for gem in env.gems]
    
    best_action = 0
    best_score = -float('inf')
    action_priority = {1: 4, 2: 3, 3: 2, 4: 1, 0: 0}  # Prefer movement over no-op
    
    for action in range(5):
        if action == 0:
            next_pos = current_pos
        elif action == 1:
            next_pos = [current_pos[0], max(0, current_pos[1]-1)]
        elif action == 2:
            next_pos = [current_pos[0], min(7, current_pos[1]+1)]
        elif action == 3:
            next_pos = [max(0, current_pos[0]-1), current_pos[1]]
        else:
            next_pos = [min(7, current_pos[0]+1), current_pos[1]]
        
        # Skip if position is dangerous
        if tuple(next_pos) in dangerous_positions:
            continue
            
        # Calculate minimum distance to any gem
        min_dist = min(abs(next_pos[0]-g[0]) + abs(next_pos[1]-g[1]) for g in gems_pos)
        score = -min_dist  # Negative because lower distance is better
        
        # Break ties using action priority
        if score > best_score or (score == best_score and action_priority[action] > action_priority[best_action]):
            best_score = score
            best_action = action
            
    return [best_action, 0, 0]