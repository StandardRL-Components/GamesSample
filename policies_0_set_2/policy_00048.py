def policy(env):
    """
    Strategy: Leverage perfect grid knowledge to always match pairs optimally. 
    If one tile is revealed, find and reveal its hidden match. Otherwise, 
    reveal the first hidden tile of a known pair. This minimizes moves by 
    avoiding mistakes and leveraging complete information.
    """
    grid = env.grid
    cursor_r, cursor_c = env.cursor_pos
    revealed_tiles = [t for t in grid if t['state'] == 'revealed']
    
    if len(revealed_tiles) == 1:
        revealed_id = revealed_tiles[0]['id']
        for idx, tile in enumerate(grid):
            if tile['state'] == 'hidden' and tile['id'] == revealed_id:
                target_r = idx // 4
                target_c = idx % 4
                if cursor_r == target_r and cursor_c == target_c:
                    return [0, 1, 0]
                row_diff = (target_r - cursor_r) % 4
                if row_diff > 2:
                    row_diff -= 4
                if row_diff != 0:
                    return [2 if row_diff > 0 else 1, 0, 0]
                col_diff = (target_c - cursor_c) % 4
                if col_diff > 2:
                    col_diff -= 4
                return [4 if col_diff > 0 else 3, 0, 0]
                
    hidden_by_id = {}
    for idx, tile in enumerate(grid):
        if tile['state'] == 'hidden':
            hidden_by_id.setdefault(tile['id'], []).append(idx)
            
    for id_val, indices in hidden_by_id.items():
        if len(indices) >= 2:
            target_idx = indices[0]
            target_r = target_idx // 4
            target_c = target_idx % 4
            if cursor_r == target_r and cursor_c == target_c:
                return [0, 1, 0]
            row_diff = (target_r - cursor_r) % 4
            if row_diff > 2:
                row_diff -= 4
            if row_diff != 0:
                return [2 if row_diff > 0 else 1, 0, 0]
            col_diff = (target_c - cursor_c) % 4
            if col_diff > 2:
                col_diff -= 4
            return [4 if col_diff > 0 else 3, 0, 0]
            
    return [0, 0, 0]