def policy(env):
    # Strategy: Maximize ore collection by mining only when an enemy is within 120 pixels (risky mining gives +5 reward vs -2 for safe mining).
    # Avoid enemies by evading when they are too close (<50 pixels), otherwise move toward the nearest asteroid with ore.
    # Use squared distance comparisons to avoid sqrt for efficiency and precision.
    player_pos = env.player_pos
    asteroids = [a for a in env.asteroids if a['active'] and a['ore'] > 0]
    enemies = env.enemies
    
    min_sq_enemy_dist = float('inf')
    for e in enemies:
        dx = player_pos.x - e['pos'].x
        dy = player_pos.y - e['pos'].y
        sq_dist = dx*dx + dy*dy
        if sq_dist < min_sq_enemy_dist:
            min_sq_enemy_dist = sq_dist

    if min_sq_enemy_dist < 2500:  # 50²
        nearest_enemy = min(enemies, key=lambda e: (player_pos.x - e['pos'].x)**2 + (player_pos.y - e['pos'].y)**2)
        dx = player_pos.x - nearest_enemy['pos'].x
        dy = player_pos.y - nearest_enemy['pos'].y
        if abs(dx) > abs(dy):
            movement = 4 if dx > 0 else 3
        else:
            movement = 2 if dy > 0 else 1
    else:
        if asteroids:
            nearest_asteroid = min(asteroids, key=lambda a: (player_pos.x - a['pos'].x)**2 + (player_pos.y - a['pos'].y)**2)
            dx = nearest_asteroid['pos'].x - player_pos.x
            dy = nearest_asteroid['pos'].y - player_pos.y
            if abs(dx) > abs(dy):
                movement = 4 if dx > 0 else 3
            else:
                movement = 2 if dy > 0 else 1
        else:
            movement = 0

    mining_possible = any(
        a['active'] and a['ore'] > 0 and (player_pos.x - a['pos'].x)**2 + (player_pos.y - a['pos'].y)**2 < 3600
        for a in env.asteroids
    )
    space_held = 1 if (min_sq_enemy_dist < 14400 and min_sq_enemy_dist >= 2500 and mining_possible) else 0

    return [movement, space_held, 0]