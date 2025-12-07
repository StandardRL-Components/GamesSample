def policy(env):
    """
    This policy uses a greedy heuristic to choose the push direction that minimizes the total Manhattan distance of blocks to their goals.
    It prioritizes moves that reduce distance, breaks ties by direction order (up, down, left, right), and returns no-op only if no moves are possible.
    """
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    action_codes = [1, 2, 3, 4]
    best_score = float('-inf')
    best_action = 0
    block_positions = set(tuple(b['pos']) for b in env.blocks)
    
    for idx, d in enumerate(directions):
        if not env._can_any_block_move(d):
            continue
        score = 0
        for block in env.blocks:
            pos, goal = tuple(block['pos']), block['goal_pos']
            curr_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            new_pos = (pos[0] + d[0], pos[1] + d[1])
            if (0 <= new_pos[0] < env.GRID_WIDTH and 
                0 <= new_pos[1] < env.GRID_HEIGHT and 
                new_pos not in block_positions):
                new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
                score += (curr_dist - new_dist)
        if score > best_score:
            best_score = score
            best_action = action_codes[idx]
            
    return [best_action, 0, 0] if best_score > float('-inf') else [0, 0, 0]