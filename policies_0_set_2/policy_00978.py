def policy(env):
    # Strategy: Prioritize destroying aliens for immediate rewards while avoiding enemy projectiles.
    # Align horizontally with the lowest alien to maximize hit chance, and fire when cooldown allows.
    # Avoid vertical movement to stay near bottom for better evasion and targeting.
    a0 = 0  # Default to no movement
    a1 = 1 if env.player_fire_timer == 0 and len(env.aliens) > 0 else 0  # Fire if ready and aliens exist
    a2 = 0  # Secondary action unused

    # Find lowest alien (highest y-value) to target
    lowest_alien = None
    for alien in env.aliens:
        if lowest_alien is None or alien['pos'][1] > lowest_alien['pos'][1]:
            lowest_alien = alien

    # Move horizontally to align with lowest alien if present
    if lowest_alien is not None:
        dx = lowest_alien['pos'][0] - env.player_pos[0]
        if abs(dx) > 10:  # Avoid jitter with tolerance
            a0 = 4 if dx > 0 else 3  # Move right if alien is to the right, else left

    # Evade nearby enemy projectiles by moving away
    for proj in env.enemy_projectiles:
        if abs(proj[0] - env.player_pos[0]) < 20 and env.player_pos[1] - proj[1] < 50:
            a0 = 3 if proj[0] > env.player_pos[0] else 4  # Move away from projectile

    return [a0, a1, a2]