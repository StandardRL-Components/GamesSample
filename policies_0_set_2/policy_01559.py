def policy(env):
    """
    Greedy policy that moves towards the nearest gem using Manhattan distance.
    Prioritizes immediate gem collection (+1 reward) and minimizes moves to maximize
    the chance of collecting all gems within the move limit. Secondary actions are
    unused in this environment.
    """
    if env.game_over or len(env.gems) == 0:
        return [0, 0, 0]
    
    robot_x, robot_y = env.robot_pos
    best_move = 0
    best_dist = float('inf')
    
    for gem_x, gem_y in env.gems:
        dist = abs(gem_x - robot_x) + abs(gem_y - robot_y)
        if dist < best_dist:
            best_dist = dist
            dx = gem_x - robot_x
            dy = gem_y - robot_y
            
            if abs(dx) > abs(dy):
                best_move = 4 if dx > 0 else 3
            else:
                best_move = 2 if dy > 0 else 1
    
    return [best_move, 0, 0]