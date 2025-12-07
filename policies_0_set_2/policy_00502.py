def policy(env):
    # Strategy: Maximize reward by destroying aliens while avoiding projectiles.
    # Always fire when possible (a1=1) to destroy aliens for +1 reward per alien.
    # Move horizontally to dodge incoming projectiles (priority) and center under alien formations to maximize shot efficiency.
    # Secondary action (a2) unused in this game, so set to 0.
    player_x, _ = env.player_pos
    enemy_projectiles = env.enemy_projectiles
    aliens = env.aliens
    
    movement = 0  # Default: no movement
    min_dist_sq = 10000  # 100^2 threshold for projectile avoidance
    closest_proj = None
    
    # Find closest projectile within avoidance radius
    for proj in enemy_projectiles:
        dx = proj[0] - player_x
        dy = proj[1] - env.player_pos[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_proj = proj
            
    # Dodge closest projectile by moving horizontally away
    if closest_proj is not None and min_dist_sq < 10000:
        dx = closest_proj[0] - player_x
        movement = 4 if dx < 0 else 3  # Move right if projectile left, else left
    else:
        # No immediate threat: move toward average alien x-position to optimize shooting
        if aliens:
            total_x = sum(alien['pos'][0] for alien in aliens)
            avg_x = total_x / len(aliens)
            if avg_x < player_x - 10:
                movement = 3  # Move left
            elif avg_x > player_x + 10:
                movement = 4  # Move right
                
    return [movement, 1, 0]