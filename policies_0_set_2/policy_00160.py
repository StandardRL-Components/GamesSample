def policy(env):
    # Strategy: Avoid obstacles by aligning with gap centers, use boost when available, and move right for progress.
    # This maximizes reward by ensuring continuous forward movement while minimizing collisions.
    
    # Check boost availability
    boost_action = 1 if env.boost_cooldown_left == 0 and env.boost_steps_left == 0 else 0
    
    # Find nearest obstacle ahead
    player_world_x = env.camera_x + env.player_pos.x
    threshold = 200
    next_obstacle_x = None
    for obs in env.obstacles:
        if obs.x > player_world_x:
            if next_obstacle_x is None or obs.x < next_obstacle_x:
                next_obstacle_x = obs.x
    
    # Adjust vertical position if obstacle is near
    if next_obstacle_x is not None and (next_obstacle_x - player_world_x) < threshold:
        top_obstacle = None
        bottom_obstacle = None
        for obs in env.obstacles:
            if obs.x == next_obstacle_x:
                if obs.y == 0:
                    top_obstacle = obs
                else:
                    bottom_obstacle = obs
        
        if top_obstacle and bottom_obstacle:
            gap_center = (top_obstacle.height + bottom_obstacle.y) / 2
        elif top_obstacle:
            gap_center = (top_obstacle.height + env.HEIGHT) / 2
        elif bottom_obstacle:
            gap_center = bottom_obstacle.y / 2
        else:
            gap_center = env.HEIGHT / 2
        
        if env.player_pos.y < gap_center - 10:
            movement = 2  # Move down
        elif env.player_pos.y > gap_center + 10:
            movement = 1  # Move up
        else:
            movement = 4  # Move right if aligned
    else:
        movement = 4  # Move right if no obstacle near
    
    return [movement, boost_action, 0]