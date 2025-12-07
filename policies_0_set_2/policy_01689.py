def policy(env):
    # Strategy: Extend current path greedily until no more moves, then clear to maximize line reward (length + 10 bonus).
    # If path is too short, reset with shift to find a better start. Avoids penalties by always having a valid action.
    if env.game_over:
        return [0, 0, 0]
    if not env.drag_path:
        return [0, 0, 1]
    x, y = env.cursor_pos
    color = env.grid[env.drag_path[0][1]][env.drag_path[0][0]]
    for idx, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)], start=1):
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.GRID_COLS and 0 <= ny < env.GRID_ROWS:
            if env.grid[ny][nx] == color and (nx, ny) not in env.drag_path:
                return [idx, 0, 0]
    return [0, 1, 0] if len(env.drag_path) >= 2 else [0, 0, 1]