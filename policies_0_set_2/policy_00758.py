def policy(env):
    """
    Maximizes mining efficiency by navigating to nearest mineral-rich asteroid, 
    aligning the ship for optimal mining beam use, and avoiding collisions. 
    Prioritizes immediate mineral collection to reach the 50-mineral goal before time runs out.
    """
    import math
    import numpy as np
    
    # Find nearest asteroid with minerals
    min_dist = float('inf')
    target_asteroid = None
    for asteroid in env.asteroids:
        if asteroid['minerals'] > 0:
            dist = np.linalg.norm(env.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                target_asteroid = asteroid
    
    if target_asteroid is None:
        return [0, 0, 0]  # No valid targets
    
    # Calculate direction to target
    direction = target_asteroid['pos'] - env.player_pos
    target_angle = math.atan2(direction[1], direction[0])
    
    # Calculate angle difference (normalized to [-π, π])
    angle_diff = (target_angle - env.player_angle + math.pi) % (2 * math.pi) - math.pi
    
    # Determine movement action
    if abs(angle_diff) > 0.2:  # Need to turn
        a0 = 4 if angle_diff > 0 else 3  # Right or left turn
    elif min_dist > 100:  # Need to accelerate toward target
        a0 = 1  # Accelerate
    elif np.linalg.norm(env.player_vel) > 2:  # Need to brake
        a0 = 2  # Brake
    else:
        a0 = 0  # No movement needed
    
    # Activate mining beam when aligned and in range
    a1 = 1 if (min_dist <= env.MINING_RANGE and abs(angle_diff) < 0.3) else 0
    
    return [a0, a1, 0]  # a2 unused in this environment