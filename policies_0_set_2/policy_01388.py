def policy(env):
    """
    Maximizes reward by prioritizing survival (avoiding projectiles) while aggressively targeting enemies.
    Always fires when cooldown allows, moves toward nearest enemy when safe, and evades nearby projectiles.
    """
    # Always set secondary action to 0 (Shift has no effect)
    a2 = 0
    
    # Always fire when cooldown allows (primary action)
    a1 = 1 if env.player_fire_cooldown == 0 else 0
    
    # Default to no movement
    a0 = 0
    
    # Get current player position
    player_pos = env.player_pos
    
    # Find nearest enemy if any exist
    nearest_enemy = None
    min_enemy_dist = float('inf')
    for enemy in env.enemies:
        dist = (enemy['pos'] - player_pos).length()
        if dist < min_enemy_dist:
            min_enemy_dist = dist
            nearest_enemy = enemy
    
    # Find nearest projectile if any exist
    nearest_proj = None
    min_proj_dist = float('inf')
    for proj in env.enemy_projectiles:
        dist = (proj['pos'] - player_pos).length()
        if dist < min_proj_dist:
            min_proj_dist = dist
            nearest_proj = proj
    
    # Evade nearby projectiles (priority)
    if nearest_proj and min_proj_dist < 50:
        evade_dir = player_pos - nearest_proj['pos']
        if abs(evade_dir.x) > abs(evade_dir.y):
            a0 = 4 if evade_dir.x > 0 else 3
        else:
            a0 = 2 if evade_dir.y > 0 else 1
    # Move toward nearest enemy when safe
    elif nearest_enemy:
        to_enemy = nearest_enemy['pos'] - player_pos
        if abs(to_enemy.x) > abs(to_enemy.y):
            a0 = 4 if to_enemy.x > 0 else 3
        else:
            a0 = 2 if to_enemy.y > 0 else 1
    
    return [a0, a1, a2]