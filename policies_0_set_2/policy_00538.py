def policy(env):
    # This policy prioritizes connecting the largest available component by first moving to a block in the largest cluster,
    # selecting it, then moving to an adjacent block in the same component to connect. It maximizes reward by clearing
    # larger groups which yield higher points and can trigger cascading clears via gravity.
    grid = env.grid
    cursor_r, cursor_c = env.cursor_pos
    selected_pos = env.selected_pos

    if selected_pos is not None:
        sel_r, sel_c = selected_pos
        if grid[sel_r, sel_c] == 0:
            return [0, 0, 1]
        color = grid[sel_r, sel_c]
        adjacents = []
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            r, c = sel_r + dr, sel_c + dc
            if 0 <= r < env.GRID_ROWS and 0 <= c < env.GRID_COLS and grid[r, c] == color:
                adjacents.append((r, c))
        if (cursor_r, cursor_c) in adjacents:
            return [0, 1, 0]
        if adjacents:
            target = min(adjacents, key=lambda pos: abs(pos[0]-cursor_r) + abs(pos[1]-cursor_c))
            dr = target[0] - cursor_r
            dc = target[1] - cursor_c
            if dr != 0:
                move = 2 if dr > 0 else 1
            else:
                move = 4 if dc > 0 else 3
            return [move, 0, 0]
        return [0, 0, 1]

    best_size = 0
    best_pos = None
    visited = set()
    for r in range(env.GRID_ROWS):
        for c in range(env.GRID_COLS):
            if (r, c) in visited or grid[r, c] == 0:
                continue
            component, color = env._find_connected_component([r, c])
            size = len(component)
            for pos in component:
                visited.add(pos)
            if size < 2:
                continue
            if size > best_size:
                best_size = size
                best_pos = (r, c)
            elif size == best_size:
                curr_dist = abs(r - cursor_r) + abs(c - cursor_c)
                best_dist = abs(best_pos[0] - cursor_r) + abs(best_pos[1] - cursor_c)
                if curr_dist < best_dist:
                    best_pos = (r, c)

    if best_pos is not None:
        t_r, t_c = best_pos
        if cursor_r == t_r and cursor_c == t_c:
            return [0, 1, 0]
        dr = t_r - cursor_r
        dc = t_c - cursor_c
        if dr != 0:
            move = 2 if dr > 0 else 1
        else:
            move = 4 if dc > 0 else 3
        return [move, 0, 0]

    return [0, 0, 0]