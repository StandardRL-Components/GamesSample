def policy(env):
    # Strategy: Maximize tower coverage by building on sites with most enemy traffic.
    # Prioritize advanced towers when affordable (wave>=3), otherwise build basic.
    # Move cursor to best available site when current site is occupied or insufficient resources.
    current_idx = env.cursor_index
    current_site = env.build_sites[current_idx]
    
    # Check if we can build at current position
    if current_site['tower'] is None:
        if env.wave_number >= 3 and env.resources >= 225:
            return [0, 0, 1]  # Build advanced tower
        elif env.resources >= 100:
            return [0, 1, 0]  # Build basic tower
    
    # Find best free build site based on enemy proximity
    best_site = None
    best_score = -1
    for idx, site in enumerate(env.build_sites):
        if site['tower'] is not None:
            continue
        # Score site by number of enemies within basic tower range
        score = 0
        site_pos = site['pos']
        for enemy in env.enemies:
            dx = enemy['pos'][0] - site_pos[0]
            dy = enemy['pos'][1] - site_pos[1]
            if dx*dx + dy*dy <= 6400:  # 80^2 (basic tower range)
                score += 1
        if score > best_score:
            best_score = score
            best_site = idx
    
    # Move to best site if found and not current position
    if best_site is not None and best_site != current_idx:
        current_row, current_col = current_idx // 2, current_idx % 2
        target_row, target_col = best_site // 2, best_site % 2
        if current_row > target_row:
            return [1, 0, 0]  # Move up
        elif current_row < target_row:
            return [2, 0, 0]  # Move down
        elif current_col > target_col:
            return [3, 0, 0]  # Move left
        else:
            return [4, 0, 0]  # Move right
    
    return [0, 0, 0]  # No action