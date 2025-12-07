def policy(env):
    """
    Maximizes reward by efficiently completing the pixel art pattern:
    1. First paints correct cells with current color when possible
    2. Moves to nearest cell needing current color when available
    3. Changes color only when no current-color cells remain
    4. Breaks movement ties consistently (right/down preferred)
    """
    cursor_x, cursor_y = env.cursor_pos
    current_color = env.current_color_index
    player_grid = env.player_grid
    target_grid = env.target_grid

    # Paint immediately if current cell is wrong and color matches
    if player_grid[cursor_y, cursor_x] != target_grid[cursor_y, cursor_x]:
        if current_color == target_grid[cursor_y, cursor_x]:
            return [0, 1, 0]

    # Find all cells needing current color
    target_cells = []
    for y in range(10):
        for x in range(10):
            if (player_grid[y, x] != target_grid[y, x] and 
                target_grid[y, x] == current_color):
                target_cells.append((x, y))
                
    if target_cells:
        # Find nearest target cell using Manhattan distance
        min_dist = float('inf')
        best_cell = None
        for x, y in target_cells:
            dist = abs(x - cursor_x) + abs(y - cursor_y)
            if dist < min_dist:
                min_dist = dist
                best_cell = (x, y)
        # Move toward best cell
        dx = best_cell[0] - cursor_x
        dy = best_cell[1] - cursor_y
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        elif dy != 0:
            return [2 if dy > 0 else 1, 0, 0]

    # Change color if no current-color targets remain
    return [0, 0, 1]