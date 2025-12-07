def policy(env):
    """
    Maximizes score by prioritizing mining the closest asteroid within range while avoiding collisions.
    If no asteroid is mineable, moves towards the closest one. Avoids movement when invulnerable or already in mining range.
    """
    a1 = 0
    # Check if any asteroid is within mining range and has minerals
    for asteroid in env.asteroids:
        if asteroid['minerals'] <= 0:
            continue
        dist_sq = (env.player_pos.x - asteroid['pos'].x)**2 + (env.player_pos.y - asteroid['pos'].y)**2
        mining_dist = env.MINING_RANGE + asteroid['radius']
        if dist_sq < mining_dist**2:
            a1 = 1
            break

    a0 = 0
    # Avoid collisions if not invulnerable
    if env.player_invulnerable_timer == 0:
        for asteroid in env.asteroids:
            dist_sq = (env.player_pos.x - asteroid['pos'].x)**2 + (env.player_pos.y - asteroid['pos'].y)**2
            collision_dist = env.PLAYER_RADIUS + asteroid['radius']
            if dist_sq < collision_dist**2:
                dx = env.player_pos.x - asteroid['pos'].x
                dy = env.player_pos.y - asteroid['pos'].y
                if abs(dx) > abs(dy):
                    a0 = 4 if dx > 0 else 3
                else:
                    a0 = 2 if dy > 0 else 1
                return [a0, a1, 0]

    # Find closest asteroid with minerals
    closest_dist_sq = float('inf')
    closest_asteroid = None
    for asteroid in env.asteroids:
        if asteroid['minerals'] <= 0:
            continue
        dist_sq = (env.player_pos.x - asteroid['pos'].x)**2 + (env.player_pos.y - asteroid['pos'].y)**2
        if dist_sq < closest_dist_sq:
            closest_dist_sq = dist_sq
            closest_asteroid = asteroid

    if closest_asteroid is None:
        return [0, a1, 0]

    # Move towards closest asteroid if not in mining range
    mining_dist = env.MINING_RANGE + closest_asteroid['radius']
    if closest_dist_sq >= mining_dist**2:
        dx = closest_asteroid['pos'].x - env.player_pos.x
        dy = closest_asteroid['pos'].y - env.player_pos.y
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1

    return [a0, a1, 0]