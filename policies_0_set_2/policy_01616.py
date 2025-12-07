def policy(env):
    """
    Maximizes score by always popping the largest available fruit group. If current cursor
    position has a valid group (size>=3), pops it. Otherwise, moves towards the largest
    group (prioritizing size then proximity) to set up future pops. This strategy efficiently
    maximizes reward per action by leveraging group size squared scoring and minimizes wasted
    moves by targeting the most valuable groups first.
    """
    cursor_x, cursor_y = env.cursor_pos
    grid = env.grid
    width, height = grid.shape[0], grid.shape[1]
    
    # Check if current position has valid group
    if grid[cursor_x, cursor_y] != 0:
        visited = set()
        stack = [(cursor_x, cursor_y)]
        visited.add((cursor_x, cursor_y))
        current_group = []
        while stack:
            x, y = stack.pop()
            current_group.append((x, y))
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and grid[nx, ny] == grid[cursor_x, cursor_y]:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        if len(current_group) >= 3:
            return [0, 1, 0]
    
    # Find all valid groups
    visited = set()
    groups = []
    for x in range(width):
        for y in range(height):
            if (x, y) not in visited and grid[x, y] != 0:
                stack = [(x, y)]
                visited.add((x, y))
                group = []
                while stack:
                    i, j = stack.pop()
                    group.append((i, j))
                    for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < width and 0 <= nj < height and (ni, nj) not in visited and grid[ni, nj] == grid[x, y]:
                            visited.add((ni, nj))
                            stack.append((ni, nj))
                if len(group) >= 3:
                    groups.append(group)
    
    if not groups:
        return [4, 0, 0]  # Default move if no groups found
    
    # Select largest group (break ties by proximity)
    best_group = None
    best_size = -1
    best_dist = float('inf')
    for group in groups:
        size = len(group)
        min_dist = min(abs(cursor_x - x) + abs(cursor_y - y) for (x, y) in group)
        if size > best_size or (size == best_size and min_dist < best_dist):
            best_size = size
            best_dist = min_dist
            best_group = group
    
    # Move toward closest cell in best group
    target = min(best_group, key=lambda p: abs(cursor_x - p[0]) + abs(cursor_y - p[1]))
    dx = target[0] - cursor_x
    dy = target[1] - cursor_y
    
    if abs(dx) > abs(dy):
        action0 = 4 if dx > 0 else 3
    else:
        action0 = 2 if dy > 0 else 1
    
    return [action0, 0, 0]