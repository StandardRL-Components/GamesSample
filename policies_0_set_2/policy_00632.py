def policy(env):
    # Strategy: Maximize survival by kiting zombies while attacking when possible.
    # Prioritize attacking with units off cooldown, maintain safe distance from zombies,
    # and switch to units that can attack or are in better position.
    if not env.player_units:
        return [0, 0, 0]
    
    selected_unit = env.player_units[env.selected_unit_idx]
    
    # Check if current unit can attack
    if selected_unit["cooldown"] == 0:
        for zombie in env.zombies:
            if selected_unit["pos"].distance_to(zombie["pos"]) <= env.PLAYER_ATTACK_RANGE:
                return [0, 0, 1]
    
    # Check if other units can attack and switch to them
    for i, unit in enumerate(env.player_units):
        if i != env.selected_unit_idx and unit["cooldown"] == 0:
            for zombie in env.zombies:
                if unit["pos"].distance_to(zombie["pos"]) <= env.PLAYER_ATTACK_RANGE:
                    return [0, 1, 0]
    
    # Movement logic: kite zombies while maintaining attack range
    nearest_zombie = None
    min_dist = float('inf')
    for zombie in env.zombies:
        dist = selected_unit["pos"].distance_to(zombie["pos"])
        if dist < min_dist:
            min_dist = dist
            nearest_zombie = zombie
    
    if nearest_zombie:
        dx = nearest_zombie["pos"].x - selected_unit["pos"].x
        dy = nearest_zombie["pos"].y - selected_unit["pos"].y
        if min_dist < 50:  # Too close - move away
            if abs(dx) > abs(dy):
                return [3 if dx > 0 else 4, 0, 0]
            else:
                return [1 if dy > 0 else 2, 0, 0]
        else:  # Maintain optimal distance
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 0, 0]
            else:
                return [2 if dy > 0 else 1, 0, 0]
    
    # No zombies - move toward center
    center_x, center_y = env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2
    dx = center_x - selected_unit["pos"].x
    dy = center_y - selected_unit["pos"].y
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]