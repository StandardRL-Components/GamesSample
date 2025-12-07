def policy(env):
    # This policy maximizes reward by efficiently completing the target image:
    # 1. Prioritizes filling empty cells with correct colors when available
    # 2. Uses shortest path color selection when wrong color is selected
    # 3. Moves to nearest fillable cell when current cell can't be filled
    current_r, current_c = env.cursor_pos
    current_empty = env.grid[current_r, current_c] == 10
    target_color = env.target_image[current_r, current_c]
    color_available = env.color_counts[target_color] > 0 if current_empty else False
    
    if current_empty and color_available:
        if env.selected_color_idx == target_color:
            return [0, 1, 0]  # Place correct color
        else:
            # Calculate shortest color change direction
            current_color = env.selected_color_idx
            diff_forward = (target_color - current_color) % 10
            if diff_forward <= 5:
                return [2, 0, 1]  # Move down to increase color
            else:
                return [1, 0, 1]  # Move up to decrease color
    else:
        # Find nearest fillable cell (empty with available color)
        min_dist = float('inf')
        target_cell = None
        for r in range(env.GRID_SIZE):
            for c in range(env.GRID_SIZE):
                if env.grid[r, c] == 10 and env.color_counts[env.target_image[r, c]] > 0:
                    dist = abs(r - current_r) + abs(c - current_c)
                    if dist < min_dist:
                        min_dist = dist
                        target_cell = (r, c)
        
        if target_cell:
            # Move toward target cell
            dr = target_cell[0] - current_r
            dc = target_cell[1] - current_c
            if dr < 0:
                return [1, 0, 0]  # Move up
            elif dr > 0:
                return [2, 0, 0]  # Move down
            elif dc < 0:
                return [3, 0, 0]  # Move left
            else:
                return [4, 0, 0]  # Move right
        else:
            return [0, 0, 0]  # No valid moves