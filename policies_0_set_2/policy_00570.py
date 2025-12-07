def policy(env):
    # This policy maximizes score by placing blue blocks in low-neighbor spots for bonus,
    # avoiding penalties for red blocks, and dropping instantly to save time. It evaluates
    # all possible columns for optimal placement based on immediate reward and block type.
    if env.current_block is None:
        return [0, 0, 0]
    
    current_x = env.current_block['grid_x']
    current_y = env.current_block['grid_y']
    block_type = env.current_block['type']
    best_x = current_x
    best_score = -float('inf')
    
    for x in range(env.GRID_WIDTH):
        if env.grid[current_y, x] != 0:
            continue
            
        landing_y = env.GRID_HEIGHT - 1
        for y in range(env.GRID_HEIGHT):
            if env.grid[y, x] != 0:
                landing_y = y - 1
                break
        if landing_y < 0:
            continue
            
        neighbors = env._count_neighbors(x, landing_y)
        base = 0.1
        penalty = 0.2 if neighbors >= 2 else 0
        bonus = 0.5 if (block_type == 2 and neighbors < 2) else 0
        net_reward = base - penalty + bonus
        
        if net_reward > best_score or (net_reward == best_score and abs(x - current_x) < abs(best_x - current_x)):
            best_score = net_reward
            best_x = x
            
    if best_x < current_x:
        return [3, 0, 0]
    elif best_x > current_x:
        return [4, 0, 0]
    else:
        return [2, 0, 0]