def policy(env):
    """
    Prioritizes pushing blocks toward their targets using Manhattan distance heuristics.
    Scores each possible movement by how much it reduces block-to-target distance,
    with bonuses for target placements. Avoids pushing blocks off targets or into walls.
    If no beneficial push exists, moves player toward the nearest off-target block.
    """
    player_pos = env.player_pos
    blocks = env.blocks
    targets = env.targets
    
    # Precompute which blocks are off-target
    off_target_blocks = [b for b in blocks if b['pos'] != targets[b['target_id']]['pos']]
    if not off_target_blocks:
        return [0, 0, 0]  # All blocks placed, no movement needed

    mov_vecs = {0: (0,0), 1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}
    best_action = 0
    best_score = -float('inf')
    
    for a0 in range(5):
        dx, dy = mov_vecs[a0]
        new_p = (player_pos[0] + dx, player_pos[1] + dy)
        
        # Check bounds for non-zero actions
        if a0 != 0 and not (0 <= new_p[0] < env.GRID_WIDTH and 0 <= new_p[1] < env.GRID_HEIGHT):
            continue
            
        # Check if pushing a block
        block_at_new_p = next((b for b in blocks if b['pos'] == new_p), None)
        if block_at_new_p and a0 != 0:
            new_b_pos = (new_p[0] + dx, new_p[1] + dy)
            # Check if push is valid
            if not (0 <= new_b_pos[0] < env.GRID_WIDTH and 0 <= new_b_pos[1] < env.GRID_HEIGHT):
                continue
            if any(b['pos'] == new_b_pos for b in blocks):
                continue
            # Avoid pushing blocks off targets
            if block_at_new_p['pos'] == targets[block_at_new_p['target_id']]['pos']:
                continue
                
            # Score push by distance improvement
            t_pos = targets[block_at_new_p['target_id']]['pos']
            old_dist = abs(block_at_new_p['pos'][0] - t_pos[0]) + abs(block_at_new_p['pos'][1] - t_pos[1])
            new_dist = abs(new_b_pos[0] - t_pos[0]) + abs(new_b_pos[1] - t_pos[1])
            score = (old_dist - new_dist) * 10
            if new_b_pos == t_pos:
                score += 100  # Bonus for target placement
        else:
            # Score movement toward nearest off-target block
            min_dist = min(abs(new_p[0] - b['pos'][0]) + abs(new_p[1] - b['pos'][1]) for b in off_target_blocks)
            score = -min_dist  # Closer is better

        if score > best_score:
            best_score = score
            best_action = a0

    return [best_action, 0, 0]