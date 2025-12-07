def policy(env):
    # Strategy: Navigate towards nearest target-colored orb while avoiding enemies when not invincible.
    # Prioritize immediate reward from collecting target orbs, using squared distance to avoid sqrt.
    player_pos = env.player_pos
    target_color = env.target_color
    orbs = env.orbs
    enemies = env.enemies
    invincible = env.invincibility_timer > 0
    
    # Find target orbs (fall back to any orb if none match target)
    target_orbs = [orb for orb in orbs if tuple(orb['color']) == tuple(target_color)]
    if not target_orbs:
        target_orbs = orbs
    if not target_orbs:
        return [0, 0, 0]
    
    # Find closest target orb using squared distance
    closest_orb = min(target_orbs, key=lambda orb: (player_pos.x - orb['pos'].x)**2 + (player_pos.y - orb['pos'].y)**2)
    
    # Evaluate movement actions
    moves = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    best_action = 0
    best_score = -float('inf')
    
    for action, (dx, dy) in moves.items():
        new_x = player_pos.x + dx * env.PLAYER_SPEED
        new_y = player_pos.y + dy * env.PLAYER_SPEED
        new_x = max(env.PLAYER_RADIUS, min(env.WIDTH - env.PLAYER_RADIUS, new_x))
        new_y = max(env.PLAYER_RADIUS, min(env.HEIGHT - env.PLAYER_RADIUS, new_y))
        
        # Score based on distance to target orb
        dist_sq = (new_x - closest_orb['pos'].x)**2 + (new_y - closest_orb['pos'].y)**2
        score = -dist_sq
        
        # Avoid enemies when not invincible
        if not invincible:
            for enemy in enemies:
                enemy_dist_sq = (new_x - enemy['pos'].x)**2 + (new_y - enemy['pos'].y)**2
                if enemy_dist_sq < (env.PLAYER_RADIUS + env.ENEMY_RADIUS + 10)**2:
                    score -= 1000000  # Large penalty for proximity to enemies
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return [best_action, 0, 0]