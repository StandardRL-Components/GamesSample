def policy(env):
    # Strategy: Check if level is complete and advance by pressing space. Otherwise, move cursor to a mismatched cell and push in the direction that best aligns with the target. This reduces mismatch by focusing on corrective actions.
    if hasattr(env, 'game_over') and env.game_over and hasattr(env, 'level_complete') and env.level_complete:
        return [0, 1, 0]  # Press space to advance to next level
    
    if not (hasattr(env, 'player_grid') and hasattr(env, 'target_grid') and hasattr(env, 'grid_size')):
        return [0, 0, 0]
    
    grid_size = env.grid_size
    player_grid = env.player_grid
    target_grid = env.target_grid
    
    # Find first mismatched cell
    mismatched = None
    for r in range(grid_size):
        for c in range(grid_size):
            if player_grid[r][c] != target_grid[r][c]:
                mismatched = (r, c)
                break
        if mismatched:
            break
            
    if not mismatched:
        return [0, 0, 0]  # No mismatch found
    
    target_r, target_c = mismatched
    cursor_r, cursor_c = env.cursor_pos
    
    # Move cursor towards mismatched cell
    if cursor_r < target_r:
        return [2, 0, 0]  # Move down
    elif cursor_r > target_r:
        return [1, 0, 0]  # Move up
    elif cursor_c < target_c:
        return [4, 0, 0]  # Move right
    elif cursor_c > target_c:
        return [3, 0, 0]  # Move left
    else:
        # At mismatched cell, push in direction that minimizes local mismatch
        best_dir = 0
        best_improvement = -1
        for direction in [1, 2, 3, 4]:
            temp_grid = [row[:] for row in player_grid]
            index = cursor_r if direction in [3, 4] else cursor_c
            line = [temp_grid[r][index] for r in range(grid_size)] if direction in [1, 2] else temp_grid[index][:]
            reverse = direction in [2, 4]
            pixels = [p for p in line if p is not None]
            if reverse:
                pixels.reverse()
            padding = [None] * (grid_size - len(pixels))
            new_line = (pixels + padding) if direction in [1, 3] else (padding + pixels)
            if direction in [1, 2]:
                for r in range(grid_size):
                    temp_grid[r][index] = new_line[r]
            else:
                temp_grid[index] = new_line
            improvement = sum(1 for r in range(grid_size) for c in range(grid_size) if temp_grid[r][c] == target_grid[r][c]) - sum(1 for r in range(grid_size) for c in range(grid_size) if player_grid[r][c] == target_grid[r][c])
            if improvement > best_improvement:
                best_improvement = improvement
                best_dir = direction
        return [best_dir, 1, 0]  # Push in best direction