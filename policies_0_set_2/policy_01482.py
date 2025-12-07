def policy(env):
    """
    Policy for racing game: avoid obstacles (red) and collect boosts (green). 
    Uses internal state (obstacles and boosts positions) to decide movement.
    Avoids obstacles by moving to the opposite side of the screen when they are within 200 pixels.
    Collects boosts by moving to their vertical center when they are within 200 pixels and closer than any obstacle.
    """
    # Read current player position and define detection threshold
    player_right = env.player_rect.right
    threshold = player_right + 200
    
    # Find closest obstacle within threshold
    closest_obstacle = None
    min_obs_dist = float('inf')
    for obs in env.obstacles:
        if obs['rect'].right > player_right and obs['rect'].left < threshold:
            dist = obs['rect'].left - player_right
            if dist < min_obs_dist:
                min_obs_dist = dist
                closest_obstacle = obs
    
    # Find closest boost within threshold
    closest_boost = None
    min_boost_dist = float('inf')
    for boost in env.boosts:
        if boost['rect'].right > player_right and boost['rect'].left < threshold:
            dist = boost['rect'].left - player_right
            if dist < min_boost_dist:
                min_boost_dist = dist
                closest_boost = boost
    
    # Determine target movement based on closest objects
    if closest_boost and (not closest_obstacle or min_boost_dist < min_obs_dist):
        # Prioritize boost collection if closer than obstacle
        target_y = closest_boost['rect'].centery
    elif closest_obstacle:
        # Avoid obstacle by moving to opposite track half
        if closest_obstacle['rect'].centery < env.HEIGHT / 2:
            target_y = env.TRACK_Y_BOTTOM - env.PLAYER_HEIGHT
        else:
            target_y = env.TRACK_Y_TOP + env.PLAYER_HEIGHT
    else:
        # Default to center position when no objects nearby
        target_y = env.HEIGHT / 2
    
    # Choose movement direction based on target
    if env.player_y < target_y - 5:
        a0 = 2  # Down
    elif env.player_y > target_y + 5:
        a0 = 1  # Up
    else:
        a0 = 0  # None
    
    return [a0, 0, 0]  # a1 and a2 unused in this environment