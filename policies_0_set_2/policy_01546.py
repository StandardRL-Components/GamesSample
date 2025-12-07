def policy(env):
    # Strategy: Chase the nearest fish by moving in the direction of the largest component of the displacement vector.
    # This minimizes distance to targets efficiently, maximizing catch rate and reward while avoiding unnecessary movements.
    if not hasattr(env, 'fish_list') or len(env.fish_list) == 0:
        return [0, 0, 0]
    
    net_x, net_y = env.net_pos[0], env.net_pos[1]
    min_sq_dist = float('inf')
    target_fish = None
    
    for fish in env.fish_list:
        dx = fish.pos[0] - net_x
        dy = fish.pos[1] - net_y
        sq_dist = dx*dx + dy*dy
        if sq_dist < min_sq_dist:
            min_sq_dist = sq_dist
            target_fish = fish
            
    if target_fish is None:
        return [0, 0, 0]
        
    dx = target_fish.pos[0] - net_x
    dy = target_fish.pos[1] - net_y
    
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
        
    return [movement, 0, 0]