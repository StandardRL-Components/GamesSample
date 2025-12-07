def policy(env):
    # Strategy: Prioritize attacking adjacent enemies to maximize rewards, then move towards nearest target (enemy, boss, or stairs) to progress efficiently.
    player_pos = env.player['pos']
    # Check for adjacent enemies (including diagonals) using Euclidean distance < 1.5
    for entity in env.enemies + ([env.boss] if env.boss else []):
        dx = entity['pos'][0] - player_pos[0]
        dy = entity['pos'][1] - player_pos[1]
        if dx*dx + dy*dy < 2.25:  # 1.5^2 = 2.25
            return [0, 1, 0]  # Attack if adjacent

    # Determine target priority: nearest enemy > boss > stairs
    target = None
    if env.enemies:
        # Find nearest enemy by squared distance
        min_dist = float('inf')
        for enemy in env.enemies:
            dx = enemy['pos'][0] - player_pos[0]
            dy = enemy['pos'][1] - player_pos[1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist:
                min_dist = dist_sq
                target = enemy['pos']
    elif env.boss:
        target = env.boss['pos']
    elif env.stairs_pos:
        target = env.stairs_pos
    else:
        return [0, 0, 0]  # No target

    # Evaluate move directions towards target
    best_move = 0
    best_score = float('inf')
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    for move, (dx, dy) in moves.items():
        nx, ny = player_pos[0] + dx, player_pos[1] + dy
        # Check validity: within bounds, floor tile, and not occupied
        if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
            continue
        if env.grid[ny][nx] != 0:
            continue
        occupied = any(e['pos'] == (nx, ny) for e in env.enemies) or (env.boss and env.boss['pos'] == (nx, ny))
        if occupied:
            continue
        # Score by Manhattan distance to target
        score = abs(nx - target[0]) + abs(ny - target[1])
        if score < best_score:
            best_score = score
            best_move = move

    return [best_move, 0, 0]