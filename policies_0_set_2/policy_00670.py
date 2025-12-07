def policy(env):
    # Minesweeper strategy: Use deductive logic to identify safe tiles and mines from revealed numbers.
    # Prioritize revealing safe tiles (immediate reward) and avoid mines. Move efficiently to target tiles.
    GRID_SIZE = 8
    cursor = env.cursor_pos
    revealed = env.grid_revealed
    numbers = env.grid_numbers

    # Helper to get wrapped neighbors
    def get_neighbors(x, y):
        return [((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy) != (0,0)]

    # Deduce mines and safe tiles from revealed numbers
    known_mines = set()
    safe_tiles = set()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if revealed[x,y] and numbers[x,y] > 0:
                neighbors = get_neighbors(x,y)
                hidden_neighbors = [(nx,ny) for (nx,ny) in neighbors if not revealed[nx,ny]]
                if numbers[x,y] == len(hidden_neighbors):
                    known_mines.update(hidden_neighbors)
                num_known_mines = sum(1 for pos in hidden_neighbors if pos in known_mines)
                if numbers[x,y] == num_known_mines:
                    safe_tiles.update([pos for pos in hidden_neighbors if pos not in known_mines])

    # If current tile is safe and hidden, reveal it
    if not revealed[cursor[0],cursor[1]] and (cursor[0],cursor[1]) in safe_tiles:
        return [0, 1, 0]

    # Find closest safe tile or hidden non-mine tile
    target = None
    min_dist = float('inf')
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if not revealed[x,y] and ((x,y) in safe_tiles or (x,y) not in known_mines and not safe_tiles):
                dx = min(abs(x - cursor[0]), GRID_SIZE - abs(x - cursor[0]))
                dy = min(abs(y - cursor[1]), GRID_SIZE - abs(y - cursor[1]))
                dist = dx + dy
                if dist < min_dist or (dist == min_dist and (x < target[0] or (x == target[0] and y < target[1]))):
                    min_dist = dist
                    target = (x, y)

    # Move toward target
    if target:
        tx, ty = target
        dx = (tx - cursor[0]) % GRID_SIZE
        if dx > GRID_SIZE//2:
            dx -= GRID_SIZE
        dy = (ty - cursor[1]) % GRID_SIZE
        if dy > GRID_SIZE//2:
            dy -= GRID_SIZE

        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        elif dy != 0:
            return [2 if dy > 0 else 1, 0, 0]

    return [0, 0, 0]