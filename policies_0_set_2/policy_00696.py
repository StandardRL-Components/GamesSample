def policy(env):
    # Strategy: Maximize score by prioritizing immediate correct placements for +1 reward and avoiding -0.2 penalties.
    # When current cell is incorrect and color matches, place immediately. Otherwise, cycle color or move to nearest incorrect cell.
    n = env.GRID_SIZE
    x, y = env.cursor_pos
    target_color = env.target_image[y, x]
    current_color = env.player_grid[y, x]
    selected_color = env.selected_color_idx

    if current_color != target_color:
        if selected_color == target_color:
            return [0, 1, 0]  # Place correct color
        else:
            return [0, 0, 1]  # Cycle color to match target

    incorrect_cells = []
    for i in range(n):
        for j in range(n):
            if env.player_grid[j, i] != env.target_image[j, i]:
                incorrect_cells.append((i, j))
                
    if not incorrect_cells:
        return [0, 0, 0]  # All cells correct

    def manhattan(x1, y1, x2, y2):
        dx = min(abs(x1 - x2), n - abs(x1 - x2))
        dy = min(abs(y1 - y2), n - abs(y1 - y2))
        return dx + dy

    best_dist = float('inf')
    best_cell = None
    for cell in incorrect_cells:
        tx, ty = cell
        d = manhattan(x, y, tx, ty)
        if d < best_dist:
            best_dist = d
            best_cell = cell

    tx, ty = best_cell
    dx = (tx - x) % n
    if dx > n // 2:
        dx -= n
    dy = (ty - y) % n
    if dy > n // 2:
        dy -= n

    if abs(dx) > abs(dy):
        move = 3 if dx < 0 else 4
    else:
        move = 1 if dy < 0 else 2
    return [move, 0, 0]