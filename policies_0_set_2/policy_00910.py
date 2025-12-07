def policy(env):
    # Strategy: Use Tetris AI heuristic to evaluate placements by minimizing height, holes, and bumpiness while maximizing line clears.
    # Prioritize hard drops to save time and avoid penalties. Rotate and move to optimal positions before dropping.
    if env.block_locked:
        return [0, 0, 0]
    
    def evaluate_placement(x, rotation):
        y = 0
        while env._is_valid_move(x, y + 1, rotation):
            y += 1
        temp_grid = [list(row) for row in env.grid]
        shape_coords = env.SHAPES[env.block_shape][rotation]
        for dx, dy in shape_coords:
            nx, ny = x + dx, y + dy
            if 0 <= ny < env.GRID_HEIGHT and 0 <= nx < env.GRID_WIDTH:
                temp_grid[ny][nx] = 1
        
        lines_cleared = 0
        for row in range(env.GRID_HEIGHT):
            if all(temp_grid[row][col] != 0 for col in range(env.GRID_WIDTH)):
                lines_cleared += 1
                for col in range(env.GRID_WIDTH):
                    temp_grid[row][col] = 0
        
        new_grid = [[0] * env.GRID_WIDTH for _ in range(env.GRID_HEIGHT)]
        new_row = env.GRID_HEIGHT - 1
        for row in range(env.GRID_HEIGHT - 1, -1, -1):
            if any(temp_grid[row][col] != 0 for col in range(env.GRID_WIDTH)):
                new_grid[new_row] = temp_grid[row][:]
                new_row -= 1
        
        heights = [0] * env.GRID_WIDTH
        holes = 0
        for col in range(env.GRID_WIDTH):
            found = False
            for row in range(env.GRID_HEIGHT):
                if new_grid[row][col] != 0:
                    heights[col] = env.GRID_HEIGHT - row
                    found = True
                    break
            if not found:
                heights[col] = 0
            for row in range(env.GRID_HEIGHT - heights[col], env.GRID_HEIGHT):
                if new_grid[row][col] == 0:
                    holes += 1
        
        aggregate_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(env.GRID_WIDTH - 1))
        return -0.51 * aggregate_height - 0.76 * holes - 0.36 * bumpiness + 0.18 * lines_cleared

    best_score = -10**9
    best_rotation = env.block_rotation
    best_x = env.block_x
    rotations = list(range(len(env.SHAPES[env.block_shape])))
    for rotation in rotations:
        for x in range(env.GRID_WIDTH):
            if env._is_valid_move(x, 0, rotation):
                score = evaluate_placement(x, rotation)
                if score > best_score:
                    best_score = score
                    best_rotation = rotation
                    best_x = x

    if env.block_rotation != best_rotation and env.rotate_cooldown_timer == 0:
        return [1, 0, 0]
    if env.block_x < best_x and env.move_cooldown_timer == 0 and env._is_valid_move(env.block_x + 1, env.block_y, env.block_rotation):
        return [4, 0, 0]
    if env.block_x > best_x and env.move_cooldown_timer == 0 and env._is_valid_move(env.block_x - 1, env.block_y, env.block_rotation):
        return [3, 0, 0]
    if env.block_rotation == best_rotation and env.block_x == best_x:
        return [0, 1, 0]
    return [0, 0, 0]