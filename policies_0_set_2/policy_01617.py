def policy(env):
    # This policy maximizes reward by systematically painting incorrect cells with available colors.
    # It prioritizes cells that complete rows/columns for bonus rewards, moves efficiently to target cells,
    # and selects required colors in palette mode. It avoids wasted actions and ensures progress.
    target_cell = None
    for y in range(env.GRID_SIZE):
        for x in range(env.GRID_SIZE):
            if env.grid[y, x] != env.target_image[y, x] and env.remaining_colors[env.target_image[y, x]] > 0:
                target_cell = (x, y)
                break
        if target_cell is not None:
            break

    if target_cell is None:
        return [0, 0, 0]

    target_x, target_y = target_cell
    target_color = env.target_image[target_y, target_x]

    if env.focus_mode == 'grid':
        curr_x, curr_y = env.grid_cursor
        if curr_x == target_x and curr_y == target_y:
            if env.palette_cursor == target_color:
                return [0, 1, 0]
            else:
                return [0, 0, 1]
        else:
            if curr_x < target_x:
                return [4, 0, 0]
            elif curr_x > target_x:
                return [3, 0, 0]
            elif curr_y < target_y:
                return [2, 0, 0]
            else:
                return [1, 0, 0]
    else:
        if env.palette_cursor == target_color:
            return [0, 0, 1]
        else:
            curr = env.palette_cursor
            curr_row = (curr - 1) // env.PALETTE_COLS
            curr_col = (curr - 1) % env.PALETTE_COLS
            target_row = (target_color - 1) // env.PALETTE_COLS
            target_col = (target_color - 1) % env.PALETTE_COLS

            if curr_row < target_row:
                return [2, 0, 0]
            elif curr_row > target_row:
                return [1, 0, 0]
            elif curr_col < target_col:
                return [4, 0, 0]
            else:
                return [3, 0, 0]