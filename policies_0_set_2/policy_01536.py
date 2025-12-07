def policy(env):
    # Strategy: Maximize reward by prioritizing painting correct pixels to earn +1 and complete rows/columns for +10 bonuses.
    # First, check if current cell is incorrect and selected color matches target -> paint.
    # Else, move to closest cell requiring current color, or change color if none exist.
    # If current cell is correct, move to closest incorrect cell.
    palette = [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 255, 60), (60, 255, 255), (255, 60, 255), (255, 160, 60), (240, 240, 240)]
    bg_color = (40, 45, 60)
    obs = env._get_observation()
    player_grid = [[-1] * 10 for _ in range(10)]
    target_grid = [[0] * 10 for _ in range(10)]
    for i in range(10):
        for j in range(10):
            x_p, y_p = 60 + j * 24 + 12, 90 + i * 24 + 12
            color_p = (obs[y_p, x_p, 0], obs[y_p, x_p, 1], obs[y_p, x_p, 2])
            player_grid[i][j] = -1 if color_p == bg_color else next((idx for idx, c in enumerate(palette) if color_p == c), -1)
            x_t, y_t = 340 + j * 24 + 12, 90 + i * 24 + 12
            color_t = (obs[y_t, x_t, 0], obs[y_t, x_t, 1], obs[y_t, x_t, 2])
            target_grid[i][j] = next((idx for idx, c in enumerate(palette) if color_t == c), 0)
    cursor_i, cursor_j = -1, -1
    for i in range(10):
        for j in range(10):
            x, y = 60 + j * 24, 90 + i * 24
            if (obs[y, x, 0], obs[y, x, 1], obs[y, x, 2]) == (255, 255, 0):
                cursor_i, cursor_j = i, j
                break
        if cursor_i != -1:
            break
    if cursor_i == -1:
        cursor_i, cursor_j = 0, 0
    selected_color_index = 0
    for i in range(8):
        x, y = 206 + i * 29, 350
        if (obs[y, x, 0], obs[y, x, 1], obs[y, x, 2]) == (255, 255, 255):
            selected_color_index = i
            break
    current_correct = player_grid[cursor_i][cursor_j] == target_grid[cursor_i][cursor_j]
    if not current_correct and selected_color_index == target_grid[cursor_i][cursor_j]:
        return [0, 1, 0]
    incorrect_cells = []
    for i in range(10):
        for j in range(10):
            if player_grid[i][j] != target_grid[i][j]:
                incorrect_cells.append((i, j))
    if not incorrect_cells:
        return [0, 0, 0]
    candidate_cells = [cell for cell in incorrect_cells if target_grid[cell[0]][cell[1]] == selected_color_index]
    if candidate_cells:
        best_cell = min(candidate_cells, key=lambda cell: abs(cell[0] - cursor_i) + abs(cell[1] - cursor_j))
        di, dj = best_cell[0] - cursor_i, best_cell[1] - cursor_j
        if di != 0:
            return [2 if di > 0 else 1, 0, 0]
        else:
            return [4 if dj > 0 else 3, 0, 0]
    else:
        if not current_correct:
            return [0, 0, 1]
        else:
            best_cell = min(incorrect_cells, key=lambda cell: abs(cell[0] - cursor_i) + abs(cell[1] - cursor_j))
            di, dj = best_cell[0] - cursor_i, best_cell[1] - cursor_j
            if di != 0:
                return [2 if di > 0 else 1, 0, 0]
            else:
                return [4 if dj > 0 else 3, 0, 0]