def policy(env):
    # Strategy: Prioritize clearing blocks with highest immediate reward (longest lines) from current position.
    # If no clear available, move toward the cell with best potential clear (longest contiguous same-color line).
    # Avoid invalid moves by checking cell content before clearing.
    cursor_pos = env.cursor_pos
    grid = env.grid
    cx, cy = cursor_pos
    
    # Check if current position can clear horizontally or vertically
    current_color = grid[cx, cy]
    if current_color != 0:
        # Find contiguous blocks in both directions
        horizontal = []
        for x in range(cx, -1, -1):
            if grid[x, cy] == current_color:
                horizontal.append((x, cy))
            else:
                break
        for x in range(cx + 1, env.GRID_WIDTH):
            if grid[x, cy] == current_color:
                horizontal.append((x, cy))
            else:
                break
        
        vertical = []
        for y in range(cy, -1, -1):
            if grid[cx, y] == current_color:
                vertical.append((cx, y))
            else:
                break
        for y in range(cy + 1, env.GRID_HEIGHT):
            if grid[cx, y] == current_color:
                vertical.append((cx, y))
            else:
                break
        
        # Clear if valid (at least 2 blocks)
        if len(horizontal) >= 2 and len(horizontal) >= len(vertical):
            return [0, 1, 0]  # Horizontal clear
        if len(vertical) >= 2:
            return [0, 0, 1]  # Vertical clear
    
    # Find best move toward cell with longest potential clear
    best_score = 0
    best_pos = None
    for x in range(env.GRID_WIDTH):
        for y in range(env.GRID_HEIGHT):
            if grid[x, y] == 0:
                continue
            color = grid[x, y]
            # Check horizontal potential
            h_count = 1
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < env.GRID_WIDTH and grid[nx, y] == color:
                    h_count += 1
                    nx += dx
            # Check vertical potential
            v_count = 1
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < env.GRID_HEIGHT and grid[x, ny] == color:
                    v_count += 1
                    ny += dy
            score = max(h_count, v_count)
            if score > best_score:
                best_score = score
                best_pos = (x, y)
    
    # Move toward best position if found
    if best_pos is not None:
        tx, ty = best_pos
        if cx < tx:
            return [4, 0, 0]  # Right
        elif cx > tx:
            return [3, 0, 0]  # Left
        elif cy < ty:
            return [2, 0, 0]  # Down
        elif cy > ty:
            return [1, 0, 0]  # Up
    
    return [0, 0, 0]  # Default no-op