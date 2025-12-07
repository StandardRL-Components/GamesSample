def policy(env):
    """
    Maximizes reward by prioritizing destroying the enemy tank while avoiding damage.
    Strategy: Always face enemy, maintain optimal firing distance (100-200 pixels),
    fire when aligned and cooldown allows. Movement adjusts distance while turning
    minimizes aim error. Secondary action unused (a2=0).
    """
    import math

    # Calculate vector to enemy and distance
    dx = env.enemy_pos[0] - env.player_pos[0]
    dy = env.enemy_pos[1] - env.player_pos[1]
    distance = math.sqrt(dx*dx + dy*dy)
    target_angle = math.atan2(dy, dx)
    
    # Calculate angle difference (normalized to [-π, π])
    angle_diff = target_angle - env.player_angle
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Determine movement action
    if abs(angle_diff) > 0.1:  # Prioritize aiming if not aligned
        a0 = 3 if angle_diff < 0 else 4  # Turn toward enemy
    else:
        if distance > 200:    # Move forward if too far
            a0 = 1
        elif distance < 100:  # Move backward if too close
            a0 = 2
        else:                 # Maintain position otherwise
            a0 = 0

    # Fire when aligned, cooldown ready, and ammo available
    a1 = 1 if (abs(angle_diff) < 0.2 and 
               env.player_fire_cooldown == 0 and 
               env.player_ammo > 0) else 0
    
    return [a0, a1, 0]  # a2 unused (always 0)