def policy(env):
    # Strategy: Prioritize rightward progress to reach the goal (LEVEL_END_X). Avoid obstacles by jumping vertically when
    # they are within 50 pixels horizontally. Jump right when no obstacles are near to maximize forward velocity.
    if env.is_jumping:
        return [0, 0, 0]  # No action during jump cooldown
    
    # Find nearest uncleared obstacle ahead
    nearest_obstacle = None
    min_x_dist = float('inf')
    for obs in env.obstacles:
        if not obs['cleared'] and obs['pos'][0] > env.player_pos[0]:
            x_dist = obs['pos'][0] - env.player_pos[0]
            if x_dist < min_x_dist:
                min_x_dist = x_dist
                nearest_obstacle = obs
    
    # Avoid obstacle if within 50 pixels horizontally
    if nearest_obstacle and min_x_dist < 50:
        obs_y = nearest_obstacle['pos'][1]
        # Jump away from obstacle vertically
        if obs_y < env.player_pos[1]:
            return [2, 0, 0]  # Jump down
        else:
            return [1, 0, 0]  # Jump up
    
    # Otherwise, jump right to maximize progress
    return [4, 0, 0]