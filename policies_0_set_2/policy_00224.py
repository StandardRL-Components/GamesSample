def policy(env):
    # Strategy: Prioritize attacking adjacent enemies, then move towards nearest enemy or stairs if no enemies.
    # This minimizes health loss by eliminating threats quickly and progresses through levels efficiently.
    
    import math
    
    # Check for adjacent enemies to attack
    player_pos = env.player_pos
    for enemy in env.enemies:
        if enemy['health'] <= 0:
            continue
        dist = math.hypot(enemy['pos'][0] - player_pos[0], enemy['pos'][1] - player_pos[1])
        if dist < 1.5:
            return [0, 1, 0]  # Attack adjacent enemy
    
    # Find nearest target (enemy or stairs)
    target = None
    min_dist = float('inf')
    
    # Prioritize enemies over stairs
    for enemy in env.enemies:
        if enemy['health'] <= 0:
            continue
        dist = math.hypot(enemy['pos'][0] - player_pos[0], enemy['pos'][1] - player_pos[1])
        if dist < min_dist:
            min_dist = dist
            target = enemy['pos']
    
    # If no enemies, target stairs
    if target is None and env.stairs_pos is not None:
        target = env.stairs_pos
        min_dist = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
    
    if target is None:
        return [0, 0, 0]  # No target found
    
    # Calculate movement direction towards target
    dx = target[0] - player_pos[0]
    dy = target[1] - player_pos[1]
    
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3  # Right/Left
    else:
        movement = 2 if dy > 0 else 1  # Down/Up
    
    return [movement, 0, 0]  # Move towards target