def policy(env):
    # Strategy: Prioritize moving right toward exit while shooting enemies in range. 
    # Jump to avoid projectiles when grounded. Maximizes reward by progressing through 
    # stages while eliminating threats and avoiding damage.
    action = [4, 0, 0]  # Default: move right, no shoot, no secondary
    
    # Check for immediate projectile threats
    player_x, player_y = env.player_pos
    for proj in env.enemy_projectiles:
        dx = proj['pos'][0] - player_x
        dy = proj['pos'][1] - player_y
        if abs(dx) < 50 and abs(dy) < 30:  # Projectile close to player
            if (dx > 0 and proj['dir'] == -1) or (dx < 0 and proj['dir'] == 1):  # Moving toward player
                if env.on_ground:
                    action[0] = 1  # Jump to avoid
                break
    
    # Shoot at nearest enemy in front if available
    if env.shoot_cooldown <= 0:
        for enemy in env.enemies:
            dx = enemy['pos'][0] - player_x
            if 0 < dx < 200 and abs(enemy['pos'][1] - player_y) < 50:  # Enemy in front
                action[1] = 1
                break
    
    return action