def policy(env):
    # Strategy: Avoid obstacles by moving to safest vertical position. Prioritize immediate collision avoidance,
    # then center positioning for maximum maneuverability. Use simple reactive checks for obstacles within 200px.
    if env.game_over:
        return [0, 0, 0]
    
    safe_x_margin = 200
    safe_y_margin = 25
    danger_obstacles = []
    
    for obs in env.obstacles:
        x = obs['pos'].x
        y = obs['pos'].y
        if 120 <= x <= 320 and abs(y - env.player_pos.y) < safe_y_margin:
            danger_obstacles.append(obs)
    
    if danger_obstacles:
        closest_obs = min(danger_obstacles, key=lambda o: o['pos'].x)
        if closest_obs['pos'].y < env.player_pos.y:
            return [2, 0, 0]
        else:
            return [1, 0, 0]
    else:
        if env.player_pos.y < 190:
            return [2, 0, 0]
        elif env.player_pos.y > 210:
            return [1, 0, 0]
        else:
            return [0, 0, 0]