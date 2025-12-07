def policy(env):
    # Strategy: Maximize immediate line clears and minimize future holes by evaluating all possible placements.
    # For each rotation and horizontal position, simulate dropping the piece and score the resulting grid.
    # The score prioritizes completed lines, then minimizes holes and aggregate height.
    # Choose the best placement and return actions to achieve it (rotate, move, then hard drop).
    if env.clearing_animation is not None or env.current_piece is None:
        return [0, 0, 0]
    
    current_piece = env.current_piece
    current_rot = current_piece['rotation']
    current_x = current_piece['x']
    best_score = -10**9
    best_rot = current_rot
    best_x = current_x

    def evaluate_grid(grid):
        lines_cleared = 0
        for i in range(10):
            if all(grid[i, j] != 0 for j in range(10)):
                lines_cleared += 1
            if all(grid[j, i] != 0 for j in range(10)):
                lines_cleared += 1
        holes = 0
        for x in range(10):
            found_block = False
            for y in range(10):
                if grid[x, y] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        aggregate_height = 0
        for x in range(10):
            for y in range(10):
                if grid[x, y] != 0:
                    aggregate_height += (10 - y)
                    break
        return 10 * lines_cleared - 1 * holes - 0.5 * aggregate_height

    for rotation in range(4):
        for x in range(-2, 12):
            y = 0
            while env._is_valid_position(current_piece, offset_x=x - current_x, offset_y=y, rotation=rotation - current_rot):
                y += 1
            y -= 1
            if y < 0:
                continue
            temp_grid = env.grid.copy()
            shape = env.PIECE_SHAPES[current_piece['type']][rotation]
            for dx, dy in shape:
                grid_x, grid_y = x + dx, y + dy
                if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                    temp_grid[grid_x, grid_y] = current_piece['color_idx']
            score = evaluate_grid(temp_grid)
            if score > best_score:
                best_score = score
                best_rot = rotation
                best_x = x

    if current_rot != best_rot:
        cw_steps = (best_rot - current_rot) % 4
        ccw_steps = (current_rot - best_rot) % 4
        return [1, 0, 0] if cw_steps <= ccw_steps else [2, 0, 0]
    if current_x < best_x:
        return [4, 0, 0]
    if current_x > best_x:
        return [3, 0, 0]
    return [0, 1, 0] if not env.prev_space_held else [0, 0, 0]