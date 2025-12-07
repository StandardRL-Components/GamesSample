def policy(env):
    """
    Maximizes immediate matches by simulating all possible swaps from all positions.
    Prioritizes moves that create matches, then uses a deterministic pattern to explore
    when no matches are found. Avoids invalid swaps and optimizes for gem collection.
    """
    if env.game_state != "AWAITING_INPUT":
        return [0, 0, 0]
    
    best_action = [0, 0, 0]
    best_matches = 0
    current_pos = env.selector_pos
    
    for move in range(5):
        new_pos = current_pos.copy()
        if move == 1:  # up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif move == 2:  # down
            new_pos[1] = min(env.GRID_HEIGHT - 1, new_pos[1] + 1)
        elif move == 3:  # left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif move == 4:  # right
            new_pos[0] = min(env.GRID_WIDTH - 1, new_pos[0] + 1)
        
        for swap_dir in [1, 2]:
            target = None
            if swap_dir == 1 and new_pos[0] < env.GRID_WIDTH - 1:
                target = [new_pos[0] + 1, new_pos[1]]
            elif swap_dir == 2 and new_pos[1] < env.GRID_HEIGHT - 1:
                target = [new_pos[0], new_pos[1] + 1]
            if not target:
                continue
            
            grid_copy = env.grid.copy()
            r1, c1 = new_pos[1], new_pos[0]
            r2, c2 = target[1], target[0]
            grid_copy[r1, c1], grid_copy[r2, c2] = grid_copy[r2, c2], grid_copy[r1, c1]
            
            matches = set()
            for r in range(env.GRID_HEIGHT):
                for c in range(env.GRID_WIDTH - 2):
                    if grid_copy[r, c] != 0 and grid_copy[r, c] == grid_copy[r, c+1] == grid_copy[r, c+2]:
                        matches.update([(r, c), (r, c+1), (r, c+2)])
            for c in range(env.GRID_WIDTH):
                for r in range(env.GRID_HEIGHT - 2):
                    if grid_copy[r, c] != 0 and grid_copy[r, c] == grid_copy[r+1, c] == grid_copy[r+2, c]:
                        matches.update([(r, c), (r+1, c), (r+2, c)])
            
            if len(matches) > best_matches:
                best_matches = len(matches)
                best_action = [move, 1 if swap_dir == 1 else 0, 1 if swap_dir == 2 else 0]
    
    if best_matches > 0:
        return best_action
    
    directions = [4, 2, 3, 1]
    for d in directions:
        if d == 1 and current_pos[1] > 0:
            return [1, 0, 0]
        elif d == 2 and current_pos[1] < env.GRID_HEIGHT - 1:
            return [2, 0, 0]
        elif d == 3 and current_pos[0] > 0:
            return [3, 0, 0]
        elif d == 4 and current_pos[0] < env.GRID_WIDTH - 1:
            return [4, 0, 0]
    
    return [0, 0, 0]