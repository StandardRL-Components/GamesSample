def policy(env):
    # Strategy: Maximize reward by prioritizing tower placement during inter-wave periods. 
    # Focus on building cheapest affordable towers adjacent to path nodes near the base to maximize coverage and prevent enemy advances.
    # During active waves, minimize movement to avoid disrupting tower targeting while waiting for resource accumulation from kills.
    if env.wave_in_progress:
        return [0, 0, 0]
    
    affordable_towers = [t for t in [0, 1, 2] if env.TOWER_TYPES[t]['cost'] <= env.resources]
    if not affordable_towers:
        return [0, 0, 0]
    
    cheapest_tower = min(affordable_towers, key=lambda t: env.TOWER_TYPES[t]['cost'])
    if env.selected_tower_type != cheapest_tower:
        return [0, 0, 1]
    
    occupied_set = set(tuple(t['grid_pos']) for t in env.towers)
    current_pos = tuple(env.cursor_pos)
    if current_pos not in env.path_set and current_pos not in occupied_set:
        return [0, 1, 0]
    
    candidate_positions = set()
    for node in env.path_nodes:
        x, y = node
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and (nx, ny) not in env.path_set and (nx, ny) not in occupied_set:
                candidate_positions.add((nx, ny))
                
    if not candidate_positions:
        return [0, 0, 0]
        
    base_x, base_y = env.base_pos
    best_candidate = None
    best_score = float('inf')
    for cand in candidate_positions:
        dist = abs(cand[0]-base_x) + abs(cand[1]-base_y)
        score = (dist, cand[0], cand[1])
        if score < best_score:
            best_score = score
            best_candidate = cand
            
    cx, cy = env.cursor_pos
    tx, ty = best_candidate
    dx = tx - cx
    dy = ty - cy
    
    if abs(dx) >= abs(dy):
        if dx > 0:
            return [4, 0, 0]
        elif dx < 0:
            return [3, 0, 0]
    if dy > 0:
        return [2, 0, 0]
    elif dy < 0:
        return [1, 0, 0]
    return [0, 0, 0]