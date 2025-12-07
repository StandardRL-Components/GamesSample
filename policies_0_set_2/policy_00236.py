def policy(env):
    # Strategy: For the selected block, compute the vector to its goal. Prioritize moving along the axis with the largest difference, 
    # avoiding pushing blocks that are already in goal. If the selected block is in goal, cycle to the next block.
    block = env.blocks[env.selected_block_idx]
    if block['is_in_goal']:
        return [0, 1, 0]  # Cycle to next block if current is in goal
    
    goal = next(g for g in env.goals if g['id'] == block['id'])
    dx = goal['pos'].x - block['pos'].x
    dy = goal['pos'].y - block['pos'].y
    
    candidate_dirs = []
    if dx != 0:
        candidate_dirs.append((abs(dx), 4 if dx > 0 else 3))
    if dy != 0:
        candidate_dirs.append((abs(dy), 2 if dy > 0 else 1))
    candidate_dirs.sort(reverse=True, key=lambda x: x[0])
    
    for _, dir_val in candidate_dirs:
        if dir_val == 1: new_pos = (block['pos'].x, block['pos'].y - 1)
        elif dir_val == 2: new_pos = (block['pos'].x, block['pos'].y + 1)
        elif dir_val == 3: new_pos = (block['pos'].x - 1, block['pos'].y)
        else: new_pos = (block['pos'].x + 1, block['pos'].y)
        
        if not (0 <= new_pos[0] < env.GRID_WIDTH and 0 <= new_pos[1] < env.GRID_HEIGHT):
            continue
            
        blocking_goal = any(
            b['pos'].x == new_pos[0] and b['pos'].y == new_pos[1] and b['is_in_goal']
            for b in env.blocks if b is not block
        )
        if not blocking_goal:
            return [dir_val, 0, 0]
            
    return [0, 0, 0]