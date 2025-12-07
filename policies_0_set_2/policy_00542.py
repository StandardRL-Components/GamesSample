def policy(env):
    # Strategy: Prioritize adjacent gems for immediate reward, then move towards nearest gem to minimize moves.
    # Avoid no-ops by checking boundaries. Break ties using consistent direction priority (up, down, left, right).
    if env.game_over or len(env.gems) == 0:
        return [0, 0, 0]
    
    current_pos = env.player_pos
    gems_positions = [gem['pos'] for gem in env.gems]
    
    # Check adjacent cells for gems (priority order: up, down, left, right)
    if current_pos[1] > 0 and [current_pos[0], current_pos[1]-1] in gems_positions:
        return [1, 0, 0]
    if current_pos[1] < env.GRID_HEIGHT-1 and [current_pos[0], current_pos[1]+1] in gems_positions:
        return [2, 0, 0]
    if current_pos[0] > 0 and [current_pos[0]-1, current_pos[1]] in gems_positions:
        return [3, 0, 0]
    if current_pos[0] < env.GRID_WIDTH-1 and [current_pos[0]+1, current_pos[1]] in gems_positions:
        return [4, 0, 0]
    
    # Find nearest gem by Manhattan distance
    min_dist = float('inf')
    target_gem = None
    for gem in env.gems:
        dist = abs(gem['pos'][0]-current_pos[0]) + abs(gem['pos'][1]-current_pos[1])
        if dist < min_dist:
            min_dist = dist
            target_gem = gem
    
    # Move toward nearest gem, prioritizing larger axis difference
    dx = target_gem['pos'][0] - current_pos[0]
    dy = target_gem['pos'][1] - current_pos[1]
    
    if abs(dx) > abs(dy):
        if dx > 0 and current_pos[0] < env.GRID_WIDTH-1:
            return [4, 0, 0]
        elif dx < 0 and current_pos[0] > 0:
            return [3, 0, 0]
        elif dy > 0 and current_pos[1] < env.GRID_HEIGHT-1:
            return [2, 0, 0]
        elif dy < 0 and current_pos[1] > 0:
            return [1, 0, 0]
    else:
        if dy > 0 and current_pos[1] < env.GRID_HEIGHT-1:
            return [2, 0, 0]
        elif dy < 0 and current_pos[1] > 0:
            return [1, 0, 0]
        elif dx > 0 and current_pos[0] < env.GRID_WIDTH-1:
            return [4, 0, 0]
        elif dx < 0 and current_pos[0] > 0:
            return [3, 0, 0]
    
    return [0, 0, 0]