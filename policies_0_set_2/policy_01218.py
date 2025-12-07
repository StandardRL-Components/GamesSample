def policy(env):
    # Strategy: Prioritize placing advanced towers on path-adjacent cells for maximum coverage and damage.
    # If insufficient funds, place basic towers. Move cursor to the best free candidate cell when unable to build.
    # Candidate cells are sorted by proximity to base (right side) to protect against leaks in later waves.
    candidates = [
        (15,2), (15,6), (14,2), (14,6), (13,2), (13,6), (12,1), (12,2), (12,3),
        (12,5), (12,6), (12,7), (11,2), (11,6), (8,2), (8,7), (7,1), (7,2), (7,3),
        (7,6), (7,7), (7,8), (6,2), (6,7), (3,4), (3,7), (2,3), (2,4), (2,5),
        (2,6), (2,7), (2,8), (1,4), (1,7)
    ]
    x, y = env.cursor_pos
    if env.grid[y][x] is None:
        if env.money >= env.TOWER_ADV_COST:
            return [0, 0, 1]
        elif env.money >= env.TOWER_BASIC_COST:
            return [0, 1, 0]
    target = None
    for cand in candidates:
        cx, cy = cand
        if env.grid[cy][cx] is None:
            target = cand
            break
    if target is None:
        return [0, 0, 0]
    dx = target[0] - x
    dy = target[1] - y
    if dx > 0:
        return [4, 0, 0]
    elif dx < 0:
        return [3, 0, 0]
    if dy > 0:
        return [2, 0, 0]
    elif dy < 0:
        return [1, 0, 0]
    return [0, 0, 0]