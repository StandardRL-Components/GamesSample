def policy(env):
    """
    Maximizes reward by prioritizing shooting enemies (a1=1 when cooldown allows) and dodging projectiles.
    Moves horizontally to align with enemies above player for efficient shooting. Avoids unnecessary jumps
    to maintain stability and uses deterministic threat assessment to break ties.
    """
    # Always shoot when cooldown allows (primary action)
    a1 = 1 if env.player_shoot_timer <= 0 else 0
    a2 = 0  # Secondary action unused in this game

    # Check for immediate projectile threats
    player_x, player_y = env.player_pos
    threat_threshold = 50
    for proj in env.enemy_projectiles:
        dx = abs(proj['pos'][0] - player_x)
        dy = abs(proj['pos'][1] - player_y)
        if dx < threat_threshold and dy < threat_threshold:
            if proj['pos'][0] < player_x:
                return [4, a1, a2]  # Move right to dodge
            else:
                return [3, a1, a2]  # Move left to dodge

    # Target nearest enemy above player
    min_dist = float('inf')
    target_x = None
    for enemy in env.enemies:
        if enemy['pos'][1] < player_y:  # Enemy is above player
            dist = abs(enemy['pos'][0] - player_x)
            if dist < min_dist:
                min_dist = dist
                target_x = enemy['pos'][0]

    # Move toward target enemy if found
    if target_x is not None:
        if target_x < player_x - 10:
            return [3, a1, a2]  # Move left
        elif target_x > player_x + 10:
            return [4, a1, a2]  # Move right
        else:
            return [0, a1, a2]  # Stay aligned

    # Default: minimal movement to avoid penalties
    return [0, a1, a2]