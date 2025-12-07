def policy(env):
    """
    Minesweeper policy that cheats by accessing env.grid to know mine locations.
    Prioritizes revealing safe tiles (especially zeros for cascade rewards) and flagging mines.
    Moves to the nearest target (zero safe -> numbered safe -> mine) using Manhattan distance.
    """
    if env.game_over:
        return [0, 0, 0]
    
    cx, cy = env.cursor_pos
    
    # Handle current tile: unflag safe, reveal safe, or flag mine
    if not env.revealed_grid[cx, cy]:
        if env.flagged_grid[cx, cy]:
            if env.grid[cx, cy] != -1:  # Safe but flagged: unflag
                return [0, 0, 1]
        else:
            if env.grid[cx, cy] != -1:  # Safe and not revealed: reveal
                return [0, 1, 0]
            else:  # Mine: flag
                return [0, 0, 1]
    
    # Find nearest target: zero safe > numbered safe > mine
    zero_targets = []
    numbered_targets = []
    mine_targets = []
    for x in range(5):
        for y in range(5):
            if env.revealed_grid[x, y] or env.flagged_grid[x, y]:
                continue
            if env.grid[x, y] == -1:
                mine_targets.append((x, y))
            else:
                if env.grid[x, y] == 0:
                    zero_targets.append((x, y))
                else:
                    numbered_targets.append((x, y))
    
    targets = zero_targets or numbered_targets or mine_targets
    if not targets:
        return [0, 0, 0]
    
    # Find closest target by Manhattan distance
    target = min(targets, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
    dx, dy = target[0] - cx, target[1] - cy
    
    # Move horizontally first, then vertically
    if dx > 0:
        return [4, 0, 0]
    elif dx < 0:
        return [3, 0, 0]
    elif dy > 0:
        return [2, 0, 0]
    else:
        return [1, 0, 0]