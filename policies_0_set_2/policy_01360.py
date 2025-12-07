def policy(env):
    # Minesweeper strategy: prioritize revealing safe tiles to maximize information gain and avoid mines.
    # Use deduction to flag mines and reveal safe tiles based on adjacent numbers, then explore hidden tiles.
    # Avoid flagging unless certain to minimize penalties and conserve flags for confirmed mines.
    info = env._get_info()
    cursor_r, cursor_c = info['cursor_pos']
    if env.game_over:
        return [0, 0, 0]
    
    rows, cols = env.grid_rows, env.grid_cols
    revealed, flagged, numbers = env.revealed_grid, env.flagged_grid, env.number_grid
    
    safe_to_reveal = set()
    mines_to_flag = set()
    
    for r in range(rows):
        for c in range(cols):
            if revealed[r, c] and numbers[r, c] > 0:
                hidden_neighbors = []
                flag_count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not revealed[nr, nc]:
                                if flagged[nr, nc]:
                                    flag_count += 1
                                else:
                                    hidden_neighbors.append((nr, nc))
                n = numbers[r, c]
                if flag_count == n and hidden_neighbors:
                    safe_to_reveal.update(hidden_neighbors)
                if len(hidden_neighbors) > 0 and flag_count + len(hidden_neighbors) == n:
                    mines_to_flag.update(hidden_neighbors)
    
    def manhattan_dist(a, b):
        ar, ac = a
        br, bc = b
        dr = min(abs(ar - br), rows - abs(ar - br))
        dc = min(abs(ac - bc), cols - abs(ac - bc))
        return dr + dc
    
    def get_move(cur, tgt):
        cr, cc = cur
        tr, tc = tgt
        dr = (tr - cr) % rows
        dr = dr if dr <= rows//2 else dr - rows
        dc = (tc - cc) % cols
        dc = dc if dc <= cols//2 else dc - cols
        if abs(dr) > abs(dc):
            return 1 if dr < 0 else 2
        else:
            return 3 if dc < 0 else 4
    
    targets = []
    if safe_to_reveal:
        targets = sorted(safe_to_reveal, key=lambda t: manhattan_dist((cursor_r, cursor_c), t))
    elif mines_to_flag and info['flags_placed'] < env.NUM_MINES:
        targets = sorted(mines_to_flag, key=lambda t: manhattan_dist((cursor_r, cursor_c), t))
    else:
        for r in range(rows):
            for c in range(cols):
                if not revealed[r, c] and not flagged[r, c]:
                    targets.append((r, c))
        targets.sort(key=lambda t: manhattan_dist((cursor_r, cursor_c), t))
    
    if not targets:
        return [0, 0, 0]
    
    target_r, target_c = targets[0]
    if (cursor_r, cursor_c) == (target_r, target_c):
        if (target_r, target_c) in mines_to_flag and info['flags_placed'] < env.NUM_MINES:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    else:
        move = get_move((cursor_r, cursor_c), (target_r, target_c))
        return [move, 0, 0]