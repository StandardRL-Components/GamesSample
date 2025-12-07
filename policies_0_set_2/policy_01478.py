def policy(env):
    """
    Maximizes reward by moving towards the nearest boost using Euclidean distance minimization.
    Evaluates all movement actions to select the one that minimizes distance to the boost after movement.
    Uses squared distance to avoid sqrt and clips positions to screen boundaries. Stays still if already within collection range.
    Sets a1 and a2 to 0 as they are unused in this environment.
    """
    player = env.player_pos
    boost = env.boost_pos
    if boost is None:
        return [0, 0, 0]
    
    current_dx = player[0] - boost[0]
    current_dy = player[1] - boost[1]
    current_sqdist = current_dx * current_dx + current_dy * current_dy
    if current_sqdist < 625:
        return [0, 0, 0]
    
    best_action = 0
    best_sqdist = current_sqdist
    speed = 5
    width, height = env.WIDTH, env.HEIGHT
    
    for action_id in [1, 2, 3, 4]:
        new_x, new_y = player[0], player[1]
        if action_id == 1:
            new_y -= speed
        elif action_id == 2:
            new_y += speed
        elif action_id == 3:
            new_x -= speed
        elif action_id == 4:
            new_x += speed
        
        new_x = max(0, min(width, new_x))
        new_y = max(0, min(height, new_y))
        dx = new_x - boost[0]
        dy = new_y - boost[1]
        new_sqdist = dx * dx + dy * dy
        
        if new_sqdist < best_sqdist:
            best_sqdist = new_sqdist
            best_action = action_id
    
    return [best_action, 0, 0]