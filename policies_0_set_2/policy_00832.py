def policy(env):
    """
    Strategy: Use BFS to find the shortest path to the nearest target (exit or unactivated mechanism).
    Prioritize targets by distance, then exit over mechanisms, then position for tie-breaking.
    This minimizes move cost while ensuring mechanisms are activated to open paths to the exit.
    """
    # Get current state
    current_pos = (env.crystal_pos[0], env.crystal_pos[1])
    exit_pos = (env.exit_pos[0], env.exit_pos[1])
    
    # Collect targets: exit and unactivated mechanisms not at current position
    targets = []
    if current_pos != exit_pos:
        targets.append((exit_pos, 0))  # type 0: exit
    
    for num, mech in env.mechanisms.items():
        mech_pos = (mech['pos'][0], mech['pos'][1])
        if not mech['active'] and current_pos != mech_pos:
            targets.append((mech_pos, 1))  # type 1: mechanism
    
    if not targets:
        return [0, 0, 0]
    
    # Build traversability grid (avoid walls and closed doors)
    traversable = []
    for r in range(env.grid_height):
        row = []
        for c in range(env.grid_width):
            tile = env.grid[r][c]
            row.append(tile != 'W' and not tile.startswith('D'))
        traversable.append(row)
    
    # BFS initialization
    dist = [[-1] * env.grid_width for _ in range(env.grid_height)]
    first_move = [[0] * env.grid_width for _ in range(env.grid_height)]
    queue = [current_pos]
    dist[current_pos[1]][current_pos[0]] = 0
    moves = [1, 2, 3, 4]
    move_vectors = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    # BFS to compute distances and first moves
    while queue:
        x, y = queue.pop(0)
        for move in moves:
            dx, dy = move_vectors[move]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < env.grid_width and 0 <= ny < env.grid_height):
                continue
            if not traversable[ny][nx]:
                continue
            if dist[ny][nx] == -1:
                dist[ny][nx] = dist[y][x] + 1
                if (x, y) == current_pos:
                    first_move[ny][nx] = move
                else:
                    first_move[ny][nx] = first_move[y][x]
                queue.append((nx, ny))
    
    # Evaluate reachable targets
    candidates = []
    for pos, t_type in targets:
        x, y = pos
        if dist[y][x] != -1:
            candidates.append((dist[y][x], t_type, pos, first_move[y][x]))
    
    if not candidates:
        return [0, 0, 0]
    
    # Select best target: min distance, then exit (0) before mechanism (1), then row/column for ties
    candidates.sort(key=lambda x: (x[0], x[1], x[2][1], x[2][0]))
    best_move = candidates[0][3]
    return [best_move, 0, 0]