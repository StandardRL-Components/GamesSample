def policy(env):
    # Systematically scan the screen in a raster pattern to find all hidden objects.
    # Moves to cell centers every 0.5 seconds and clicks if unvisited, ensuring full coverage
    # while avoiding repeat clicks. This maximizes reward by finding objects efficiently
    # within the time limit while minimizing penalties from incorrect clicks.
    if env.game_over:
        return [0, 0, 0]
    cell_size = 30
    n_cols = (env.WIDTH + cell_size - 1) // cell_size
    n_rows = (env.HEIGHT + cell_size - 1) // cell_size
    total_cells = n_cols * n_rows
    time_per_cell = 0.5
    cell_index = int(env.time_elapsed / time_per_cell) % total_cells
    i = cell_index % n_cols
    j = cell_index // n_cols
    target_x = max(0, min(env.WIDTH - 1, i * cell_size + cell_size // 2))
    target_y = max(0, min(env.HEIGHT - 1, j * cell_size + cell_size // 2))
    dx = target_x - env.cursor_pos[0]
    dy = target_y - env.cursor_pos[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq < 25:
        if not env.clicked_map[int(target_x), int(target_y)]:
            return [0, 1, 0]
        else:
            return [0, 0, 0]
    else:
        if abs(dx) > abs(dy):
            if dx > 0:
                return [4, 0, 0]
            else:
                return [3, 0, 0]
        else:
            if dy > 0:
                return [2, 0, 0]
            else:
                return [1, 0, 0]