def policy(env):
    # Strategy: Jump only when obstacles are near to avoid penalties, prioritize large jumps for high bonuses
    # Use pixel color detection in critical regions to decide between small/large jumps or no action
    obs = env._get_observation()
    ground_y = env.GROUND_Y
    player_x = env.PLAYER_X_POS
    player_radius = env.PLAYER_RADIUS
    
    # Check if player is on ground (blue pixel at expected position)
    on_ground = np.any(obs[ground_y-5:ground_y+5, player_x, 2] > 200)
    if not on_ground:
        return [0, 0, 0]  # No jump while airborne
    
    # Define detection regions (obstacles: red, bonuses: yellow)
    obstacle_region = obs[ground_y-30:ground_y, player_x+player_radius:player_x+player_radius+188]
    bonus_region = obs[0:ground_y-10, player_x+player_radius:player_x+player_radius+188]
    
    # Check for obstacles (red pixels)
    red_mask = (obstacle_region[:,:,0] > 150) & (obstacle_region[:,:,1] < 100) & (obstacle_region[:,:,2] < 100)
    has_obstacle = np.any(red_mask)
    
    # Check for bonuses (yellow pixels)
    yellow_mask = (bonus_region[:,:,0] > 200) & (bonus_region[:,:,1] > 200) & (bonus_region[:,:,2] < 100)
    has_bonus = np.any(yellow_mask)
    
    if has_obstacle:
        return [0, 1, 0]  # Small jump for obstacles
    elif has_bonus:
        return [0, 0, 1]  # Large jump for bonuses
    else:
        return [0, 0, 0]  # No action