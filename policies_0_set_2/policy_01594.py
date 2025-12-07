def policy(env):
    # Strategy: Maximize coverage and damage by placing towers on optimal path-adjacent slots.
    # Prioritize Cannon towers for cost-efficiency early, then Snipers for range and damage.
    # Move cursor to best available slot and place if affordable, else wait for resources.
    n = len(env.PLACEMENT_SLOTS)
    available_slots = [i for i in range(n) if env.PLACEMENT_SLOTS[i] not in [t['pos'] for t in env.towers]]
    if not available_slots:
        return [0, 0, 0]
    
    affordable_towers = [t for t, spec in env.TOWER_SPECS.items() if spec['cost'] <= env.gold]
    if not affordable_towers:
        return [0, 0, 0]
    
    best_score = -1
    best_slot = None
    best_tower_type = None
    for slot_idx in available_slots:
        pos = env.PLACEMENT_SLOTS[slot_idx]
        for tower_type in affordable_towers:
            spec = env.TOWER_SPECS[tower_type]
            coverage = 0
            for path_pos in env.PATH:
                dx, dy = pos[0] - path_pos[0], pos[1] - path_pos[1]
                if dx*dx + dy*dy <= spec['range']**2:
                    coverage += 1
            if coverage > best_score:
                best_score = coverage
                best_slot = slot_idx
                best_tower_type = tower_type
    
    if env.selected_tower_type != best_tower_type:
        return [0, 0, 1]
    
    if env.cursor_pos_idx == best_slot:
        return [0, 1, 0]
    
    current = env.cursor_pos_idx
    target = best_slot
    clockwise = (target - current) % n
    counter_clockwise = (current - target) % n
    moves = [1, 2, 3, 4]
    new_indices = [
        (current - 1) % n,
        (current + 1) % n,
        (current - 4) % n,
        (current + 4) % n
    ]
    best_dist = float('inf')
    best_move = 0
    for i, new_idx in enumerate(new_indices):
        dist = min((new_idx - target) % n, (target - new_idx) % n)
        if dist < best_dist:
            best_dist = dist
            best_move = moves[i]
    return [best_move, 0, 0]