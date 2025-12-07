def policy(env):
    # Strategy: Maximize score by always selecting the largest available group of same-colored blocks.
    # If no group is available at the cursor, move toward the largest valid group on the board.
    # Avoid invalid moves by checking group size before selection. Break ties by proximity.
    if env.falling_blocks:
        return [0, 0, 0]
    
    grid = env.grid
    y, x = env.cursor_pos
    
    # Find all valid groups (size >= 3)
    valid_groups = []
    visited = set()
    for r in range(8):
        for c in range(8):
            if (r, c) in visited or grid[r, c] in (0, 6):
                continue
            stack = [(r, c)]
            connected = set()
            while stack:
                cy, cx = stack.pop()
                if (cy, cx) in connected:
                    continue
                connected.add((cy, cx))
                for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 8 and 0 <= nx < 8 and grid[ny, nx] == grid[r, c] and (ny, nx) not in connected:
                        stack.append((ny, nx))
            if len(connected) >= 3:
                valid_groups.append((len(connected), r, c))
            visited.update(connected)
    
    if not valid_groups:
        return [0, 0, 0]
    
    # Check current position first
    current_group_size = 0
    if grid[y, x] not in (0, 6):
        stack = [(y, x)]
        connected = set()
        while stack:
            cy, cx = stack.pop()
            if (cy, cx) in connected:
                continue
            connected.add((cy, cx))
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < 8 and 0 <= nx < 8 and grid[ny, nx] == grid[y, x] and (ny, nx) not in connected:
                    stack.append((ny, nx))
        current_group_size = len(connected)
    
    if current_group_size >= 3:
        return [0, 1, 0]
    
    # Find largest group (prioritize size, then proximity)
    valid_groups.sort(key=lambda x: (-x[0], abs(x[1] - y) + abs(x[2] - x)))
    target_r, target_c = valid_groups[0][1], valid_groups[0][2]
    
    # Move toward target
    if y < target_r and y < 7:
        return [2, 0, 0]
    elif y > target_r and y > 0:
        return [1, 0, 0]
    elif x < target_c and x < 7:
        return [4, 0, 0]
    elif x > target_c and x > 0:
        return [3, 0, 0]
    
    return [0, 0, 0]