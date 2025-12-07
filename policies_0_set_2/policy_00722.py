def policy(env):
    # Strategy: Prioritize dodging enemy projectiles to avoid damage, then attack nearest enemy when safe.
    # Move towards nearest enemy to maintain optimal attack range and update facing direction for accurate attacks.
    # Use dodge only when threatened and available to maximize survivability and reward.
    player = env.player
    enemies = env.enemies
    enemy_projectiles = env.enemy_projectiles
    
    # If dodging, do nothing until dodge completes
    if player['is_dodging']:
        return [0, 0, 0]
    
    # Check for immediate threats from projectiles
    threat = False
    nearest_proj = None
    min_proj_dist = float('inf')
    for proj in enemy_projectiles:
        dist = player['pos'].distance_to(proj['pos'])
        if dist < 50:  # Projectile is close
            to_player = player['pos'] - proj['pos']
            if to_player.length() > 0 and proj['dir'].dot(to_player.normalize()) > 0.5:  # Moving toward player
                threat = True
                if dist < min_proj_dist:
                    min_proj_dist = dist
                    nearest_proj = proj
    
    # Dodge if threatened and dodge available
    if threat and player['dodge_cooldown'] == 0:
        return [0, 0, 1]
    
    # Move away from nearest threatening projectile if cannot dodge
    if threat and nearest_proj is not None:
        away_dir = player['pos'] - nearest_proj['pos']
        if abs(away_dir.x) > abs(away_dir.y):
            movement = 4 if away_dir.x > 0 else 3
        else:
            movement = 2 if away_dir.y > 0 else 1
        return [movement, 0, 0]
    
    # If no enemies, do nothing
    if not enemies:
        return [0, 0, 0]
    
    # Find nearest enemy and move toward it
    nearest_enemy = min(enemies, key=lambda e: player['pos'].distance_to(e['pos']))
    to_enemy = nearest_enemy['pos'] - player['pos']
    if abs(to_enemy.x) > abs(to_enemy.y):
        movement = 4 if to_enemy.x > 0 else 3
    else:
        movement = 2 if to_enemy.y > 0 else 1
    
    # Attack if cooldown ready and not dodging
    attack = 1 if player['attack_cooldown'] == 0 else 0
    return [movement, attack, 0]