def policy(env):
    """
    Maximizes score by prioritizing chain reactions (>=2 gems) for bonus rewards, then single gems.
    Checks current cell first, then adjacent cells for best chain. Moves without collecting only when necessary.
    """
    grid = env.grid
    x, y = env.cursor_pos

    def get_chain_length(cx, cy):
        if not (0 <= cx < 5 and 0 <= cy < 5) or grid[cy, cx] == 0:
            return 0
        target = grid[cy, cx]
        visited = set()
        stack = [(cx, cy)]
        visited.add((cx, cy))
        while stack:
            px, py = stack.pop()
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = (px + dx) % 5, (py + dy) % 5
                if (nx, ny) not in visited and grid[ny, nx] == target:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        return len(visited)

    current_chain = get_chain_length(x, y)
    if current_chain >= 2:
        return [0, 1, 0]

    best_dir = None
    best_chain = 0
    for direction, (dx, dy) in enumerate([(0,-1), (0,1), (-1,0), (1,0)], 1):
        nx, ny = (x + dx) % 5, (y + dy) % 5
        chain = get_chain_length(nx, ny)
        if chain > best_chain:
            best_chain = chain
            best_dir = direction

    if best_chain >= 2:
        return [best_dir, 1, 0]
    elif grid[y, x] != 0:
        return [0, 1, 0]
    elif best_dir is not None:
        return [best_dir, 0, 0]
    else:
        return [4, 0, 0]