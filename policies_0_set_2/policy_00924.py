def policy(env):
    # Strategy: Stay centered in track to avoid crashes, accelerate when safe, brake near walls or sharp turns.
    # Maximizes reward by completing tracks quickly while avoiding penalties from near-misses and crashes.
    if env.game_over:
        return [0, 0, 0]
    
    wall_top, wall_bottom = env._get_walls_at(env.track_progress)
    current_center = (wall_top + wall_bottom) / 2
    error = current_center - env.player_y
    
    lookahead = 200
    wall_top_ahead, wall_bottom_ahead = env._get_walls_at(env.track_progress + lookahead)
    center_ahead = (wall_top_ahead + wall_bottom_ahead) / 2
    curvature = abs(center_ahead - current_center)
    
    min_distance = min(env.player_y - wall_top, wall_bottom - env.player_y)
    
    if error > 5:
        steer_action = 4
    elif error < -5:
        steer_action = 3
    else:
        steer_action = 0
    
    brake_condition = (min_distance < 20 and env.player_speed > 10) or (curvature > 10 and env.player_speed > 15)
    
    if brake_condition:
        return [steer_action, 0, 1]
    else:
        return [steer_action, 1, 0]