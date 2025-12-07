def policy(env):
    # Strategy: Prioritize avoiding collisions (health critical), then mine nearest ore-rich asteroid.
    # Move directly towards best target when safe, using mining beam only when in range (80 units).
    # This maximizes ore collection while minimizing health loss from collisions.
    player_x, player_y = env.player_pos
    player_radius = 12
    mining_range = 80
    avoidance_margin = 10
    
    # Check for immediate collision threats
    dangerous_asteroids = []
    for asteroid in env.asteroids:
        ast_x, ast_y = asteroid['pos']
        dist_sq = (player_x - ast_x)**2 + (player_y - ast_y)**2
        radius_sum = player_radius + asteroid['radius'] + avoidance_margin
        if dist_sq < radius_sum**2:
            dangerous_asteroids.append(asteroid)
    
    if dangerous_asteroids:
        # Avoid closest threatening asteroid
        closest = min(dangerous_asteroids, key=lambda a: (player_x - a['pos'][0])**2 + (player_y - a['pos'][1])**2)
        dx = player_x - closest['pos'][0]
        dy = player_y - closest['pos'][1]
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 0, 0]
    
    # Find best mining target (nearest with ore)
    valid_asteroids = [a for a in env.asteroids if a['ore'] > 0]
    if not valid_asteroids:
        return [0, 0, 0]
    
    target = min(valid_asteroids, key=lambda a: (player_x - a['pos'][0])**2 + (player_y - a['pos'][1])**2)
    tx, ty = target['pos']
    dist_sq = (player_x - tx)**2 + (player_y - ty)**2
    
    if dist_sq <= mining_range**2:
        # Mine while maintaining position
        dx = tx - player_x
        dy = ty - player_y
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 1, 0]
    else:
        # Move toward target
        dx = tx - player_x
        dy = ty - player_y
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 0, 0]