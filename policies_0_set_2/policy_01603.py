def policy(env):
    # Strategy: Prioritize survival by dodging enemy projectiles while continuously firing.
    # Move horizontally to align with enemies for efficient shooting and maximize score.
    # Avoid vertical movement unless necessary to dodge, maintaining optimal firing position.
    
    # Check if game over to return safe action
    if env.game_over:
        return [0, 0, 0]
    
    # Always fire when cooldown allows
    fire_action = 1 if env.player['fire_cooldown'] == 0 else 0
    
    # Get current player position
    player_pos = env.player['pos']
    
    # Find nearest enemy projectile within threat range
    threat_range = 100
    nearest_threat = None
    min_threat_dist = float('inf')
    for proj in env.enemy_projectiles:
        dist = proj['pos'].distance_to(player_pos)
        if dist < min_threat_dist and dist < threat_range:
            min_threat_dist = dist
            nearest_threat = proj
    
    # Dodge if threat is close
    if nearest_threat is not None:
        threat_vec = nearest_threat['pos'] - player_pos
        # Prefer horizontal dodging to maintain firing position
        if abs(threat_vec.x) > abs(threat_vec.y):
            return [4 if threat_vec.x > 0 else 3, fire_action, 0]
        else:
            return [2 if threat_vec.y > 0 else 1, fire_action, 0]
    
    # Find optimal horizontal alignment with enemies
    if env.enemies:
        # Calculate average x-position of enemies above player
        above_enemies = [e for e in env.enemies if e['pos'].y < player_pos.y]
        if above_enemies:
            avg_x = sum(e['pos'].x for e in above_enemies) / len(above_enemies)
            if player_pos.x < avg_x - 10:
                return [4, fire_action, 0]
            elif player_pos.x > avg_x + 10:
                return [3, fire_action, 0]
    
    # Default: minimal movement while maintaining fire
    return [0, fire_action, 0]