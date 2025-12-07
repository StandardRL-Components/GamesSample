def policy(env):
    # Strategy: Prioritize obstacle avoidance by maintaining safe vertical distance from obstacles ahead.
    # Stay near track center when safe to maximize checkpoint progress, but dynamically adjust position
    # based on obstacle proximity using a weighted risk assessment of upcoming obstacles.
    
    # Calculate risk from obstacles within 200px ahead
    risk_above = 0
    risk_below = 0
    for obstacle in env.obstacles:
        if 100 <= obstacle['rect'].x <= 300:  # Look ahead 200px
            # Calculate vertical distance to obstacle center
            vert_dist = obstacle['rect'].centery - env.car_pos.y
            risk = max(0, 1 - abs(vert_dist) / 100)  # Normalized risk [0,1]
            
            if vert_dist < 0:  # Obstacle above
                risk_above += risk
            else:  # Obstacle below
                risk_below += risk
    
    # Avoid obstacles by moving away from higher risk direction
    if risk_above > risk_below + 0.2:
        return [2, 0, 0]  # Move down
    elif risk_below > risk_above + 0.2:
        return [1, 0, 0]  # Move up
    
    # When risks are balanced, follow track center with hysteresis
    track_center = env._get_track_y_at(env.camera_x + 100)
    if env.car_pos.y < track_center - 8:
        return [2, 0, 0]  # Move down toward center
    elif env.car_pos.y > track_center + 8:
        return [1, 0, 0]  # Move up toward center
    
    return [0, 0, 0]  # Maintain position