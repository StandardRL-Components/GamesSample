def policy(env):
    # Strategy: Prioritize reaching exit by moving right, jumping over obstacles, and shooting enemies.
    # Maximizes reward by minimizing distance to exit while avoiding damage and collecting enemy kill bonuses.
    a0 = 4  # Default: move right
    a1 = 0
    a2 = 0
    
    # Check shooting cooldown
    can_shoot = (env.steps - env.player_last_shot) > env.PLAYER_SHOOT_COOLDOWN
    
    # Check for obstacles in front requiring jump
    player_right = env.player_rect.right
    for obstacle in env.obstacles:
        if (obstacle.left <= player_right + 50 and obstacle.right >= player_right and
            obstacle.bottom > env.player_rect.top and obstacle.top < env.player_rect.bottom and
            env.on_ground):
            a0 = 1  # Jump over obstacle
            break
    
    # Check for nearby enemies to shoot
    for enemy in env.enemies:
        enemy_rect = enemy["rect"]
        dx = enemy_rect.x - env.player_rect.x
        dy = abs(enemy_rect.y - env.player_rect.y)
        if dx > 0 and dx < 300 and dy < 50 and can_shoot:
            a1 = 1  # Shoot enemy
            break
    
    return [a0, a1, a2]