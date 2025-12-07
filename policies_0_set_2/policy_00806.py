def policy(env):
    # Strategy: Maximize survival and dodge rewards by staying centered when safe, 
    # moving away from imminent obstacles towards the largest gap, and using boost 
    # for rapid evasion while avoiding boundaries and oscillation.
    player_y = env.player_y
    player_vy = env.player_vy
    obstacles = env.obstacles
    TRACK_Y_TOP = env.TRACK_Y_TOP
    TRACK_Y_BOTTOM = env.TRACK_Y_BOTTOM
    PLAYER_HEIGHT = env.PLAYER_HEIGHT
    PLAYER_X = env.PLAYER_X
    
    lookahead = 150
    closest_obstacle = None
    min_dist = float('inf')
    
    for obs in obstacles:
        if obs['passed']:
            continue
        dist = obs['rect'].left - PLAYER_X
        if 0 < dist < lookahead and dist < min_dist:
            min_dist = dist
            closest_obstacle = obs
    
    action0 = 0
    action1 = 0
    action2 = 0
    
    if closest_obstacle:
        obs_rect = closest_obstacle['rect']
        danger_zone_top = obs_rect.top - PLAYER_HEIGHT/2
        danger_zone_bottom = obs_rect.bottom + PLAYER_HEIGHT/2
        
        if danger_zone_top < player_y < danger_zone_bottom:
            space_above = obs_rect.top - TRACK_Y_TOP
            space_below = TRACK_Y_BOTTOM - obs_rect.bottom
            if space_above > space_below:
                action0 = 1
            else:
                action0 = 2
            action1 = 1
        else:
            if player_y < TRACK_Y_TOP + 30:
                action0 = 2
            elif player_y > TRACK_Y_BOTTOM - 30:
                action0 = 1
    else:
        center = (TRACK_Y_TOP + TRACK_Y_BOTTOM) / 2
        if abs(player_y - center) > 20:
            action0 = 1 if player_y > center else 2
    
    return [action0, action1, action2]