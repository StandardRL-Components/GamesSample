def policy(env):
    # Strategy: Prioritize hopping to the target planet when safe, otherwise choose the safest path to minimize asteroid collisions and maximize score.
    if env.player_state == "HOPPING":
        return [0, 0, 0]
    
    target_idx = None
    for i, p in enumerate(env.planets):
        if p['type'] == 'target':
            target_idx = i
            break
            
    if target_idx is None:
        return [0, 0, 0]
        
    n_planets = len(env.sorted_planet_indices)
    if n_planets == 0:
        return [0, 0, 0]
    
    if target_idx in env.sorted_planet_indices:
        target_sorted_idx = env.sorted_planet_indices.index(target_idx)
        if not env.planets[target_idx]['is_risky'] or env.lives > 1:
            desired_idx = target_sorted_idx
        else:
            safe_dist = float('inf')
            desired_idx = 0
            for i, idx in enumerate(env.sorted_planet_indices):
                if not env.planets[idx]['is_risky']:
                    dx = env.planets[idx]['pos'][0] - env.planets[target_idx]['pos'][0]
                    dy = env.planets[idx]['pos'][1] - env.planets[target_idx]['pos'][1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < safe_dist:
                        safe_dist = dist_sq
                        desired_idx = i
    else:
        min_dist = float('inf')
        desired_idx = 0
        for i, idx in enumerate(env.sorted_planet_indices):
            dx = env.planets[idx]['pos'][0] - env.planets[target_idx]['pos'][0]
            dy = env.planets[idx]['pos'][1] - env.planets[target_idx]['pos'][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist:
                min_dist = dist_sq
                desired_idx = i
                
    if env.selector_idx == desired_idx:
        return [0, 1, 0]
        
    n = n_planets
    cw_steps = (desired_idx - env.selector_idx) % n
    ccw_steps = (env.selector_idx - desired_idx) % n
    
    if cw_steps <= ccw_steps:
        if env.last_movement_action != 1:
            return [1, 0, 0]
        else:
            return [4, 0, 0]
    else:
        if env.last_movement_action != 3:
            return [3, 0, 0]
        else:
            return [2, 0, 0]