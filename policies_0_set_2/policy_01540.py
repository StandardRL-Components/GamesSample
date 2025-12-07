def policy(env):
    # Minesweeper strategy: use revealed numbers to infer safe tiles and mines, prioritizing safe reveals for flood-fill rewards and avoiding mine hits. Move to inferred safe tiles first, then least risky hidden tiles based on adjacent numbers.
    if env.game_over or env.win:
        return [0, 0, 0]
    
    def wrapped_diff(a, b, size):
        diff = (b - a) % size
        return diff if diff <= size // 2 else diff - size

    def get_move(current, target, size):
        dr = wrapped_diff(current[0], target[0], size)
        dc = wrapped_diff(current[1], target[1], size)
        if dr != 0:
            return 2 if dr > 0 else 1
        elif dc != 0:
            return 4 if dc > 0 else 3
        else:
            return 0

    safe_tiles = set()
    mine_tiles = set()
    size = env.GRID_SIZE
    current = env.cursor_pos
    
    for r in range(size):
        for c in range(size):
            if env.grid_state[r, c] == 1 and env.adjacency_map[r, c] > 0:
                hidden_neighbors = []
                flagged_neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            state = env.grid_state[nr, nc]
                            if state == 0:
                                hidden_neighbors.append((nr, nc))
                            elif state == 2:
                                flagged_neighbors.append((nr, nc))
                n = env.adjacency_map[r, c]
                if len(flagged_neighbors) == n and hidden_neighbors:
                    safe_tiles.update(hidden_neighbors)
                elif len(flagged_neighbors) + len(hidden_neighbors) == n and hidden_neighbors:
                    mine_tiles.update(hidden_neighbors)
    
    r, c = current
    if (r, c) in safe_tiles:
        return [0, 1, 0]
    if (r, c) in mine_tiles:
        return [0, 0, 1]
    
    if safe_tiles:
        target = min(safe_tiles, key=lambda t: abs(wrapped_diff(t[0], current[0], size)) + abs(wrapped_diff(t[1], current[1], size)))
        move = get_move(current, target, size)
        return [move, 0, 0]
    
    if mine_tiles:
        target = min(mine_tiles, key=lambda t: abs(wrapped_diff(t[0], current[0], size)) + abs(wrapped_diff(t[1], current[1], size)))
        move = get_move(current, target, size)
        return [move, 0, 0]
    
    hidden_tiles = [(i, j) for i in range(size) for j in range(size) if env.grid_state[i, j] == 0]
    if not hidden_tiles:
        return [0, 0, 0]
    
    def calculate_risk(tile):
        r, c = tile
        risk = 0.0
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and env.grid_state[nr, nc] == 1 and env.adjacency_map[nr, nc] > 0:
                    hidden_count = sum(1 for dr2 in [-1,0,1] for dc2 in [-1,0,1] if not (dr2==0 and dc2==0) and 0<=nr+dr2<size and 0<=nc+dc2<size and env.grid_state[nr+dr2, nc+dc2]==0)
                    if hidden_count > 0:
                        risk += env.adjacency_map[nr, nc] / hidden_count
                    count += 1
        return risk / count if count > 0 else 0.123
    
    best_tile = min(hidden_tiles, key=lambda t: (calculate_risk(t), abs(wrapped_diff(t[0], current[0], size)) + abs(wrapped_diff(t[1], current[1], size))))
    move = get_move(current, best_tile, size)
    return [move, 0, 0]