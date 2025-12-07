def policy(env):
    # Strategy: Jump only when on ground and an obstacle is within 100 pixels ahead. Use high jump for obstacles requiring >80 clearance, else small jump.
    if not env.on_ground:
        return [0, 0, 0]
    
    lookahead = 100
    closest_obstacle = None
    min_distance = float('inf')
    for obstacle in env.obstacles:
        obstacle_screen_x = obstacle["pos"][0] - (env.player_world_x - 120)
        if obstacle_screen_x + obstacle["size"][0] > 120 and obstacle_screen_x < 120 + lookahead:
            distance = obstacle_screen_x - 120
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle
                
    if closest_obstacle is not None:
        clearance_needed = env.FLOOR_Y - closest_obstacle["pos"][1]
        if clearance_needed > 80:
            return [0, 1, 1]
        else:
            return [0, 1, 0]
            
    return [0, 0, 0]