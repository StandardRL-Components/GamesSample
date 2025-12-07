def policy(env):
    """
    Strategy: Avoid traps at all costs while moving towards the nearest gem using Manhattan distance.
    Prioritize immediate gem collection, then move to minimize distance to remaining gems.
    Secondary actions (a1, a2) are unused in this environment and set to 0.
    """
    player_pos = env.player_pos
    gems = env.gems
    traps = env.traps
    
    if not gems:
        return [0, 0, 0]
    
    def manhattan_dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    safe_actions = []
    for action in [0, 1, 2, 3, 4]:
        new_pos = player_pos.copy()
        if action == 1:
            new_pos[1] -= 1
        elif action == 2:
            new_pos[1] += 1
        elif action == 3:
            new_pos[0] -= 1
        elif action == 4:
            new_pos[0] += 1
        
        new_pos[0] = max(0, min(env.GRID_WIDTH - 1, new_pos[0]))
        new_pos[1] = max(0, min(env.GRID_HEIGHT - 1, new_pos[1]))
        
        if new_pos not in traps:
            safe_actions.append((action, new_pos))
    
    if not safe_actions:
        return [0, 0, 0]
    
    best_action = None
    min_dist = float('inf')
    for action, pos in safe_actions:
        dist = min(manhattan_dist(pos, gem) for gem in gems)
        if dist < min_dist:
            min_dist = dist
            best_action = action
    
    return [best_action, 0, 0]