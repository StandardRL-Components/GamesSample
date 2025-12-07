def policy(env):
    # Greedily move towards nearest gem to maximize collection rate given limited moves (10 moves to collect 20 gems)
    px, py = env.player_pos
    gems = env.gems
    
    if not gems:
        return [4, 0, 0]  # No gems left, move right arbitrarily
    
    # Find nearest gem using Manhattan distance with consistent tie-breaking
    min_dist = float('inf')
    target_gem = None
    for gem in gems:
        gx, gy = gem
        dist = abs(px - gx) + abs(py - gy)
        if dist < min_dist or (dist == min_dist and (gx < target_gem[0] or (gx == target_gem[0] and gy < target_gem[1]))):
            min_dist = dist
            target_gem = gem

    gx, gy = target_gem
    dx, dy = gx - px, gy - py
    
    # Move in direction with largest component difference to minimize moves
    if abs(dx) > abs(dy):
        action0 = 3 if dx < 0 else 4  # left or right
    else:
        action0 = 1 if dy < 0 else 2  # up or down
        
    return [action0, 0, 0]