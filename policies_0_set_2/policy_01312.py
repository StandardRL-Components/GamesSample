def policy(env):
    # Strategy: Maximize immediate rewards by selecting largest available matches first.
    # Moves cursor toward the center of the largest valid group (≥3 blocks) to trigger cascades.
    # Avoids invalid moves and prioritizes harvesting existing matches over creating new ones.
    def find_component(grid, r0, c0):
        color = grid[r0, c0]
        if color == 0:
            return set()
        comp = set()
        stack = [(r0, c0)]
        visited = set(stack)
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < env.GRID_SIZE and 0 <= nc < env.GRID_SIZE and 
                    (nr, nc) not in visited and grid[nr, nc] == color):
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return comp

    grid, (r, c) = env.grid, env.cursor_pos
    if grid[r, c] != 0:
        comp = find_component(grid, r, c)
        if len(comp) >= 3:
            return [0, 1, 0]

    best_group, best_size = None, 0
    visited = set()
    for i in range(env.GRID_SIZE):
        for j in range(env.GRID_SIZE):
            if (i, j) in visited or grid[i, j] == 0:
                continue
            comp = find_component(grid, i, j)
            visited.update(comp)
            if len(comp) > best_size:
                best_group, best_size = comp, len(comp)

    if best_group is None:
        return [0, 0, 0]

    tr, tc = min(best_group)
    dr = (tr - r) % env.GRID_SIZE
    if dr > env.GRID_SIZE//2:
        dr -= env.GRID_SIZE
    dc = (tc - c) % env.GRID_SIZE
    if dc > env.GRID_SIZE//2:
        dc -= env.GRID_SIZE

    if dr != 0:
        return [1 if dr < 0 else 2, 0, 0]
    if dc != 0:
        return [3 if dc < 0 else 4, 0, 0]
    return [0, 1, 0]