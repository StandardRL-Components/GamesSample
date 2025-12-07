def policy(env):
    # Strategy: Use internal state to deduce safe moves and mines. Prioritize revealing safe cells (especially zeros for flood fill) and flagging mines. Move cursor to the nearest deduced cell, breaking ties by row-major order.
    cursor_r, cursor_c = env.cursor_pos
    grid_size = env.GRID_SIZE
    safe_cells = set()
    mine_cells = set()
    
    # Find safe cells and mine cells by analyzing adjacent numbers and flags
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            if env.revealed_mask[r, c] and env.adjacent_counts[r, c] > 0:
                adj_unrevealed = []
                adj_flagged = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
                            if not env.revealed_mask[nr, nc]:
                                if env.flagged_mask[nr, nc]:
                                    adj_flagged += 1
                                else:
                                    adj_unrevealed.append((nr, nc))
                n = env.adjacent_counts[r, c]
                if adj_flagged == n and adj_unrevealed:
                    safe_cells.update(adj_unrevealed)
                if adj_flagged + len(adj_unrevealed) == n and adj_unrevealed:
                    mine_cells.update(adj_unrevealed)
    
    # Determine target cell: safe first, then mine, then first unrevealed non-flagged
    target = None
    if safe_cells:
        target = min(safe_cells)
    elif mine_cells:
        target = min(mine_cells)
    else:
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                if not env.revealed_mask[r, c] and not env.flagged_mask[r, c]:
                    target = (r, c)
                    break
            if target is not None:
                break
        if target is None:
            return [0, 0, 0]
    
    # If cursor is on target, take action
    if (cursor_r, cursor_c) == target:
        if target in safe_cells or (target not in mine_cells and not env.flagged_mask[target[0], target[1]]):
            return [0, 1, 0]
        elif target in mine_cells and not env.flagged_mask[target[0], target[1]]:
            return [0, 0, 1]
        else:
            return [0, 0, 0]
    
    # Move cursor toward target using wrapped distance
    dr = (target[0] - cursor_r) % grid_size[0]
    if dr > grid_size[0] // 2:
        dr -= grid_size[0]
    dc = (target[1] - cursor_c) % grid_size[1]
    if dc > grid_size[1] // 2:
        dc -= grid_size[1]
    
    if abs(dr) > abs(dc):
        return [2 if dr > 0 else 1, 0, 0]
    else:
        return [4 if dc > 0 else 3, 0, 0]