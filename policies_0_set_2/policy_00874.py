def policy(env):
    """
    Maximizes reward by prioritizing painting incorrect pixels with the correct color when possible. 
    Otherwise, moves towards the nearest incorrect pixel requiring the current color to minimize color cycles. 
    If no such pixel exists, cycles color to the most common required color among incorrect pixels to reduce future cycles.
    Uses wrapped Manhattan distance with consistent tie-breaking for movement decisions.
    """
    if env.game_over:
        return [0, 0, 0]
    
    grid_dim = env.GRID_DIM
    cx, cy = env.cursor_pos
    current_color = env.selected_color_index
    player_grid = env.player_grid
    target_grid = env.target_grid
    
    # Check if current pixel is incorrect and can be painted immediately
    if player_grid[cy, cx] != target_grid[cy, cx]:
        if current_color == target_grid[cy, cx]:
            return [0, 1, 0]  # Paint
        else:
            # Find all incorrect pixels requiring current color
            current_color_targets = []
            for y in range(grid_dim):
                for x in range(grid_dim):
                    if (player_grid[y, x] != target_grid[y, x] and 
                        target_grid[y, x] == current_color):
                        current_color_targets.append((x, y))
            
            if current_color_targets:
                # Move to nearest pixel requiring current color
                best_dist = float('inf')
                best_cell = current_color_targets[0]
                for (x, y) in current_color_targets:
                    dx = min(abs(x - cx), grid_dim - abs(x - cx))
                    dy = min(abs(y - cy), grid_dim - abs(y - cy))
                    dist = dx + dy
                    if dist < best_dist or (dist == best_dist and (y, x) < (best_cell[1], best_cell[0])):
                        best_dist = dist
                        best_cell = (x, y)
                tx, ty = best_cell
                dx = (tx - cx) % grid_dim
                if dx > grid_dim // 2:
                    dx -= grid_dim
                dy = (ty - cy) % grid_dim
                if dy > grid_dim // 2:
                    dy -= grid_dim
                if abs(dx) > abs(dy):
                    movement = 4 if dx > 0 else 3
                else:
                    movement = 2 if dy > 0 else 1
                return [movement, 0, 0]
            else:
                # Cycle to most common required color among incorrect pixels
                color_counts = {}
                for y in range(grid_dim):
                    for x in range(grid_dim):
                        if player_grid[y, x] != target_grid[y, x]:
                            color = target_grid[y, x]
                            color_counts[color] = color_counts.get(color, 0) + 1
                if color_counts:
                    target_color = max(color_counts, key=color_counts.get)
                    if current_color != target_color:
                        return [0, 0, 1]  # Cycle color
                return [0, 0, 0]  # No incorrect pixels requiring action
    
    # Find nearest incorrect pixel requiring current color
    best_dist = float('inf')
    best_cell = None
    for y in range(grid_dim):
        for x in range(grid_dim):
            if (player_grid[y, x] != target_grid[y, x] and 
                target_grid[y, x] == current_color):
                dx = min(abs(x - cx), grid_dim - abs(x - cx))
                dy = min(abs(y - cy), grid_dim - abs(y - cy))
                dist = dx + dy
                if dist < best_dist or (dist == best_dist and (y, x) < (best_cell[1], best_cell[0])):
                    best_dist = dist
                    best_cell = (x, y)
    
    if best_cell is not None:
        tx, ty = best_cell
        dx = (tx - cx) % grid_dim
        if dx > grid_dim // 2:
            dx -= grid_dim
        dy = (ty - cy) % grid_dim
        if dy > grid_dim // 2:
            dy -= grid_dim
        if abs(dx) > abs(dy):
            movement = 4 if dx > 0 else 3
        else:
            movement = 2 if dy > 0 else 1
        return [movement, 0, 0]
    
    # If no incorrect pixels require current color, find any incorrect pixel
    incorrect = []
    for y in range(grid_dim):
        for x in range(grid_dim):
            if player_grid[y, x] != target_grid[y, x]:
                incorrect.append((x, y))
    
    if not incorrect:
        return [0, 0, 0]  # All correct
    
    # Move to nearest incorrect pixel
    best_dist = float('inf')
    best_cell = incorrect[0]
    for (x, y) in incorrect:
        dx = min(abs(x - cx), grid_dim - abs(x - cx))
        dy = min(abs(y - cy), grid_dim - abs(y - cy))
        dist = dx + dy
        if dist < best_dist or (dist == best_dist and (y, x) < (best_cell[1], best_cell[0])):
            best_dist = dist
            best_cell = (x, y)
    tx, ty = best_cell
    dx = (tx - cx) % grid_dim
    if dx > grid_dim // 2:
        dx -= grid_dim
    dy = (ty - cy) % grid_dim
    if dy > grid_dim // 2:
        dy -= grid_dim
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    return [movement, 0, 0]