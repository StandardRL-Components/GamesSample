def policy(env):
    """
    Minesweeper policy that maximizes reward by first flagging all unflagged mines (+5 each),
    then revealing safe unrevealed cells. Uses Manhattan distance to prioritize nearest targets.
    Avoids penalties by only flagging mines and revealing safe cells, ensuring optimal win.
    """
    if env.game_over:
        return [0, 0, 0]
    
    cx, cy = env.cursor_pos
    mines = []
    safe = []
    for y in range(env.GRID_SIZE):
        for x in range(env.GRID_SIZE):
            if env.mine_grid[y, x] and env.state_grid[y, x] != env.STATE_FLAGGED:
                mines.append((x, y))
            elif not env.mine_grid[y, x] and env.state_grid[y, x] == env.STATE_UNREVEALED:
                safe.append((x, y))
    
    target = None
    if mines:
        target = min(mines, key=lambda p: abs(p[0]-cx) + abs(p[1]-cy))
    elif safe:
        target = min(safe, key=lambda p: abs(p[0]-cx) + abs(p[1]-cy))
    else:
        return [0, 0, 0]
    
    tx, ty = target
    if cx == tx and cy == ty:
        if env.mine_grid[ty, tx]:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    
    dx = tx - cx
    dy = ty - cy
    if abs(dx) > abs(dy):
        move = 4 if dx > 0 else 3
    else:
        move = 2 if dy > 0 else 1
    return [move, 0, 0]