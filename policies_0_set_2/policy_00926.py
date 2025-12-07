def policy(env):
    # Strategy: Prioritize placing towers in high-priority zones (precomputed based on path proximity and base defense).
    # Always move to the best available free zone. Place the most affordable effective tower when energy allows.
    priority_zones = [5, 4, 3, 1, 2, 0, 7, 6, 8, 9]
    current_zone = env.cursor_zone_index
    occupied_zones = {tuple(t['pos']) for t in env.towers}
    free_zones = [idx for idx in priority_zones if env.tower_placement_zones[idx] not in occupied_zones]
    
    if not free_zones:
        return [0, 0, 0]
    
    best_zone = free_zones[0]
    if current_zone == best_zone:
        if env.energy >= env.TOWER_1_COST:
            return [0, 1, 0]
        elif env.energy >= env.TOWER_2_COST:
            return [0, 0, 1]
        else:
            return [0, 0, 0]
    else:
        current_pos = env.tower_placement_zones[current_zone]
        target_pos = env.tower_placement_zones[best_zone]
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]