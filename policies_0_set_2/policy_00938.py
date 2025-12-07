def policy(env):
    # Strategy: Accelerate upward (toward finish line) while avoiding obstacles. Prioritize forward movement
    # but steer away from nearby obstacles. Secondary actions are unused in this environment.
    player_x, player_y = env.player_pos[0], env.player_pos[1]
    
    # Check for obstacles in immediate path
    danger_obstacles = []
    for obs in env.obstacles:
        dx = obs['pos'][0] - player_x
        dy = obs['pos'][1] - player_y
        if dy < 0 and abs(dx) < 50 and abs(dy) < 100:  # Obstacle ahead and close
            danger_obstacles.append(obs)
    
    if danger_obstacles:
        # Avoid closest obstacle by steering away from its x-position
        closest_obs = min(danger_obstacles, key=lambda o: (o['pos'][0]-player_x)**2 + (o['pos'][1]-player_y)**2)
        dx = closest_obs['pos'][0] - player_x
        movement = 3 if dx > 0 else 4  # Steer left if obstacle is right, else right
    else:
        movement = 1  # Accelerate upward toward finish line
    
    return [movement, 0, 0]  # a1 and a2 unused in this environment