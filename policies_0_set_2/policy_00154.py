def policy(env):
    # Strategy: Find the largest group of same-colored blocks and move towards its center, then click to clear.
    # This maximizes points by targeting large groups and minimizes time wasted on small groups.
    # If no valid groups exist, do nothing to avoid penalties.

    # Initialize group tracking
    cell_group = [[0] * env.GRID_HEIGHT for _ in range(env.GRID_WIDTH)]
    group_map = {}
    best_size = 0
    best_center = None
    group_id = 1

    # Find all connected groups
    for x in range(env.GRID_WIDTH):
        for y in range(env.GRID_HEIGHT):
            if env.grid[x, y] != 0 and cell_group[x][y] == 0:
                group_cells = []
                queue = [(x, y)]
                cell_group[x][y] = group_id
                while queue:
                    cx, cy = queue.pop(0)
                    group_cells.append((cx, cy))
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and 
                            env.grid[nx, ny] == env.grid[x, y] and cell_group[nx][ny] == 0):
                            cell_group[nx][ny] = group_id
                            queue.append((nx, ny))
                
                size = len(group_cells)
                if size > 0:
                    avg_x = sum(cx for cx, _ in group_cells) / size
                    avg_y = sum(cy for _, cy in group_cells) / size
                    group_map[group_id] = (size, (avg_x, avg_y))
                    if size > best_size:
                        best_size = size
                        best_center = (avg_x, avg_y)
                    group_id += 1

    # Click if current cursor is on a valid group (size >= 2)
    cursor_x, cursor_y = env.cursor_pos
    if env.grid[cursor_x, cursor_y] != 0:
        gid = cell_group[cursor_x][cursor_y]
        if gid != 0:
            size, _ = group_map[gid]
            if size >= 2:
                return [0, 1, 0]

    # Move towards largest group center if exists
    if best_size >= 2:
        cur_x, cur_y = env.cursor_pos
        target_x, target_y = best_center
        if abs(target_x - cur_x) > abs(target_y - cur_y):
            if target_x > cur_x:
                return [4, 0, 0]  # Right
            else:
                return [3, 0, 0]  # Left
        else:
            if target_y > cur_y:
                return [2, 0, 0]  # Down
            else:
                return [1, 0, 0]  # Up

    return [0, 0, 0]  # No valid moves