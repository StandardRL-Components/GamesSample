def policy(env):
    # Strategy: Place blocks to create a layered defense with walls channeling enemies into turret kill zones.
    # Walls are placed in a horizontal barrier, turrets cover flanks, and slowers are centered for maximum coverage.
    # This setup forces enemies into longer paths under fire, maximizing damage and minimizing fortress hits.
    
    # If no blocks left, trigger wave start with no-op
    if sum(env.available_blocks.values()) == 0:
        return [0, 0, 0]
    
    # Determine target block type (priority: walls > turrets > slower)
    target_block_type = None
    for block_type in [1, 2, 3]:
        if env.available_blocks[block_type] > 0:
            target_block_type = block_type
            break
            
    # Cycle block type if not selected
    current_block_idx = env.selected_block_idx
    target_block_idx = target_block_type - 1  # Convert type to index (1->0, 2->1, 3->2)
    if current_block_idx != target_block_idx:
        return [0, 0, 1]  # Press shift to cycle block
    
    # Calculate desired x-coordinate based on wave (move left each wave)
    desired_x = max(0, 15 - (env.wave - 1))
    
    # Define target cells for each block type
    if target_block_type == 1:  # Walls
        desired_cells = [(desired_x, y) for y in [7, 8, 9, 10, 11]]
    elif target_block_type == 2:  # Turrets
        desired_cells = [(desired_x - 1, 5), (desired_x - 1, 13)]
    else:  # Slower
        desired_cells = [(desired_x - 1, 9)]
    
    # Find first available cell in desired cells (checking grid occupancy)
    target_cell = None
    for cell in desired_cells:
        x, y = cell
        if 0 <= x < env.GRID_COLS and 0 <= y < env.GRID_ROWS and env.grid[x, y] == 0:
            target_cell = cell
            break
            
    # If all desired cells occupied, find nearest free cell to first desired cell
    if target_cell is None:
        base_x, base_y = desired_cells[0]
        # Search nearby cells by Manhattan distance
        best_dist = float('inf')
        for dx in range(-env.GRID_COLS, env.GRID_COLS):
            for dy in range(-env.GRID_ROWS, env.GRID_ROWS):
                x, y = base_x + dx, base_y + dy
                if (0 <= x < env.GRID_COLS and 0 <= y < env.GRID_ROWS and 
                    env.grid[x, y] == 0 and abs(dx) + abs(dy) < best_dist):
                    best_dist = abs(dx) + abs(dy)
                    target_cell = (x, y)
    
    # Move cursor to target cell
    cx, cy = env.cursor_pos
    tx, ty = target_cell
    if cx < tx:
        return [4, 0, 0]  # Right
    elif cx > tx:
        return [3, 0, 0]  # Left
    elif cy < ty:
        return [2, 0, 0]  # Down
    elif cy > ty:
        return [1, 0, 0]  # Up
    else:
        return [0, 1, 0]  # Press space to place block