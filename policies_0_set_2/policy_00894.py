def policy(env):
    # Strategy: Maintain speed by accelerating, steer to avoid obstacles and stay centered in track.
    # Prioritize avoiding collisions (high penalty) and passing checkpoints (high reward).
    if env.game_over:
        return [0, 0, 0]
    
    car_world_x = env.world_scroll_x + env.PLAYER_X_POS
    lookahead_dist = min(200, env.TRACK_LENGTH - car_world_x - 1)
    lookahead_x = car_world_x + lookahead_dist
    
    idx = min(int(lookahead_x), len(env.track_top_border) - 1)
    top_y = env.track_top_border[idx][1]
    bottom_y = env.track_bottom_border[idx][1]
    track_center = (top_y + bottom_y) / 2
    
    target_y = track_center
    min_obstacle_dist = float('inf')
    closest_obstacle = None
    
    for obs in env.obstacles:
        if obs['x'] < car_world_x:
            continue
        dist = obs['x'] - car_world_x
        if dist < min_obstacle_dist:
            min_obstacle_dist = dist
            closest_obstacle = obs
    
    if closest_obstacle and min_obstacle_dist < 200:
        obs_y = closest_obstacle['y']
        if abs(obs_y - env.car_y) < 25:
            if obs_y > track_center:
                target_y = top_y + 15
            else:
                target_y = bottom_y - 15
    
    movement = 0
    if env.car_y < target_y - 5:
        movement = 4
    elif env.car_y > target_y + 5:
        movement = 3
    
    return [movement, 1, 0]