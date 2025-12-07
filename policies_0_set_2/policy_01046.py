def policy(env):
    # Strategy: Prioritize fuel collection when low to avoid game over, then maximize rightward progress.
    # Select the reachable target with highest score: fuel targets when fuel < 15 (safe threshold),
    # otherwise the rightmost asteroid. Cycle selection efficiently using angular list ordering.
    if not env.reachable_targets:
        return [0, 0, 0]
    
    current_idx = env.selected_target_idx
    if current_idx not in env.reachable_targets:
        return [4, 0, 0]
    
    player_x = env.asteroids[env.player_asteroid_idx]['pos'].x
    best_score = -float('inf')
    best_index = env.reachable_targets[0]
    
    for idx in env.reachable_targets:
        asteroid = env.asteroids[idx]
        score = asteroid['pos'].x
        if env.fuel < 15 and asteroid['has_fuel']:
            score += 1000
        if score > best_score:
            best_score = score
            best_index = idx
            
    if current_idx == best_index:
        return [0, 1, 0]
        
    current_pos = env.reachable_targets.index(current_idx)
    best_pos = env.reachable_targets.index(best_index)
    n = len(env.reachable_targets)
    clockwise = (best_pos - current_pos) % n
    if clockwise <= n - clockwise:
        return [4, 0, 0]
    else:
        return [3, 0, 0]