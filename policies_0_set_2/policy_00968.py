def policy(env):
    # Strategy: Greedily move towards nearest gem to maximize collection rate and minimize time penalty.
    # Prioritizes immediate gem collection with Manhattan distance, breaking ties by preferring special gems.
    if env.move_cooldown > 0:
        return [0, 0, 0]  # Wait during cooldown
    
    player_pos = env.player_grid_pos
    gems = env.gems
    
    if not gems:
        return [0, 0, 0]  # No gems remaining
    
    # Find closest gem using Manhattan distance
    min_dist = float('inf')
    target_gem = None
    for gem in gems:
        dist = abs(player_pos[0] - gem['grid_pos'][0]) + abs(player_pos[1] - gem['grid_pos'][1])
        if dist < min_dist or (dist == min_dist and gem['is_special']):
            min_dist = dist
            target_gem = gem['grid_pos']
    
    # Determine best movement direction
    dx = target_gem[0] - player_pos[0]
    dy = target_gem[1] - player_pos[1]
    
    if abs(dx) > abs(dy):
        if dx > 0 and player_pos[0] < env.GRID_WIDTH - 1:
            return [4, 0, 0]  # Right
        elif dx < 0 and player_pos[0] > 0:
            return [3, 0, 0]  # Left
    if dy > 0 and player_pos[1] < env.GRID_HEIGHT - 1:
        return [2, 0, 0]  # Down
    elif dy < 0 and player_pos[1] > 0:
        return [1, 0, 0]  # Up
    
    return [0, 0, 0]  # Default to no movement