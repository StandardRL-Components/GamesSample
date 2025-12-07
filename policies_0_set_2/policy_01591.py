def policy(env):
    # This policy maximizes reward by prioritizing safe crystal collection and minimizing distance to nearest crystal.
    # It predicts enemy movements to avoid collisions and breaks ties deterministically towards the crystal.
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def compute_next_pos(current, move):
        x, y = current
        if move == 1: y -= 1
        elif move == 2: y += 1
        elif move == 3: x -= 1
        elif move == 4: x += 1
        x = max(0, min(env.GRID_WIDTH - 1, x))
        y = max(0, min(env.GRID_HEIGHT - 1, y))
        return (x, y)
    
    if env.game_over:
        return [0, 0, 0]
    
    current_pos = env.player_pos
    crystals = env.crystals
    enemies = env.enemies
    
    next_enemy_positions = []
    for enemy in enemies:
        next_idx = (enemy['path_index'] + 1) % len(enemy['path'])
        next_enemy_positions.append(enemy['path'][next_idx])
    
    target = None
    min_dist = float('inf')
    for crystal in crystals:
        d = manhattan(current_pos, crystal)
        if d < min_dist:
            min_dist = d
            target = crystal
    
    safe_collect_actions = []
    safe_actions = []
    unsafe_collect_actions = []
    unsafe_actions = []
    
    for move in range(5):
        next_pos = compute_next_pos(current_pos, move)
        safe = next_pos not in next_enemy_positions
        collects = next_pos in crystals
        
        if safe and collects:
            safe_collect_actions.append(move)
        elif safe:
            safe_actions.append(move)
        elif collects:
            unsafe_collect_actions.append(move)
        else:
            unsafe_actions.append(move)
    
    if safe_collect_actions:
        return [safe_collect_actions[0], 0, 0]
    
    if safe_actions:
        best_move = safe_actions[0]
        best_dist = manhattan(compute_next_pos(current_pos, best_move), target) if target else 0
        for move in safe_actions[1:]:
            next_pos = compute_next_pos(current_pos, move)
            dist = manhattan(next_pos, target) if target else 0
            if dist < best_dist:
                best_move = move
                best_dist = dist
        return [best_move, 0, 0]
    
    if unsafe_collect_actions:
        return [unsafe_collect_actions[0], 0, 0]
    
    if unsafe_actions:
        best_move = unsafe_actions[0]
        best_dist = manhattan(compute_next_pos(current_pos, best_move), target) if target else 0
        for move in unsafe_actions[1:]:
            next_pos = compute_next_pos(current_pos, move)
            dist = manhattan(next_pos, target) if target else 0
            if dist < best_dist:
                best_move = move
                best_dist = dist
        return [best_move, 0, 0]
    
    return [0, 0, 0]