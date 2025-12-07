def policy(env):
    # Strategy: Evaluate all possible pushes for current, next, and previous blocks.
    # Prioritize moves that increase blocks on target, then moves that reduce distance to targets.
    # Use selection cycling only when no beneficial push is available for current block.
    n_blocks = len(env.blocks)
    if n_blocks == 0:
        return [0, 0, 0]
    
    current_idx = env.selected_block_idx
    dir_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    best_action = [0, 0, 0]
    best_delta = -float('inf')
    
    def compute_push_delta(block_idx, direction):
        dx, dy = dir_map[direction]
        if not env._is_push_valid(block_idx, dx, dy):
            return -float('inf')
        
        block_positions = {tuple(b['pos']): b for b in env.blocks}
        target_dict = {tuple(t['pos']): t['color'] for t in env.targets}
        current_pos = env.blocks[block_idx]['pos']
        line_positions = []
        pos = current_pos
        while tuple(pos) in block_positions:
            line_positions.append(tuple(pos))
            pos = (pos[0] + dx, pos[1] + dy)
        
        before = 0
        for p in line_positions:
            if p in target_dict and target_dict[p] == block_positions[p]['color']:
                before += 1
                
        after = 0
        for p in line_positions:
            new_p = (p[0] + dx, p[1] + dy)
            if new_p in target_dict and target_dict[new_p] == block_positions[p]['color']:
                after += 1
                
        return after - before

    for direction in range(1, 5):
        delta = compute_push_delta(current_idx, direction)
        if delta > best_delta:
            best_delta = delta
            best_action = [direction, 0, 0]
    
    next_idx = (current_idx + 1) % n_blocks
    prev_idx = (current_idx - 1) % n_blocks
    
    for direction in range(1, 5):
        for cycle_type, idx in [(1, next_idx), (2, prev_idx)]:
            delta = compute_push_delta(idx, direction)
            if delta > best_delta:
                best_delta = delta
                best_action = [direction, 1 if cycle_type == 1 else 0, 1 if cycle_type == 2 else 0]
    
    if best_delta > -float('inf'):
        return best_action
    
    if env._is_push_valid(next_idx, 0, 0):
        return [0, 1, 0]
    if env._is_push_valid(prev_idx, 0, 0):
        return [0, 0, 1]
    
    return [0, 0, 0]