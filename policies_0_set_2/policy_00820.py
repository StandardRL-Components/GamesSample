def policy(env):
    """
    Maximizes reward by prioritizing immediate matches to clear tiles and gain points.
    Simulates all possible move+swap actions to find the one yielding the most matches.
    If no matches found, moves cursor in a fixed pattern to explore the board.
    Avoids invalid swaps and forfeiting to prevent penalties.
    """
    best_action = None
    best_score = -1
    
    for move in range(5):
        for swap in [0, 1]:
            if move == 0 and swap == 0:
                continue
                
            grid_copy = env.grid.copy()
            cursor_pos = env.cursor_pos
            last_move_dir = env.last_move_dir
            
            new_cursor = list(cursor_pos)
            new_last_move_dir = last_move_dir
            if move != 0:
                x, y = cursor_pos
                if move == 1:
                    y = (y - 1) % env.GRID_SIZE
                    new_last_move_dir = (0, -1)
                elif move == 2:
                    y = (y + 1) % env.GRID_SIZE
                    new_last_move_dir = (0, 1)
                elif move == 3:
                    x = (x - 1) % env.GRID_SIZE
                    new_last_move_dir = (-1, 0)
                elif move == 4:
                    x = (x + 1) % env.GRID_SIZE
                    new_last_move_dir = (1, 0)
                new_cursor = [x, y]
            
            if swap:
                p1_r, p1_c = new_cursor
                p2_r = p1_r + new_last_move_dir[1]
                p2_c = p1_c + new_last_move_dir[0]
                if not (0 <= p2_r < env.GRID_SIZE and 0 <= p2_c < env.GRID_SIZE):
                    continue
                grid_copy[p1_r, p1_c], grid_copy[p2_r, p2_c] = grid_copy[p2_r, p2_c], grid_copy[p1_r, p1_c]
            
            matches = env._find_matches(grid_copy)
            score = len(matches)
            
            if score > best_score:
                best_score = score
                best_action = [move, swap, 0]
    
    if best_score > 0:
        return best_action
        
    directions = [4, 2, 3, 1]
    move_dir = directions[env.steps % 4]
    return [move_dir, 0, 0]