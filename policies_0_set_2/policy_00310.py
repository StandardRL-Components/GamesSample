def policy(env):
    # This policy uses a heuristic-based approach to maximize line clears by evaluating potential placements
    # for the current piece. It considers all rotations and horizontal positions, simulating drops to assess
    # the resulting grid state based on aggregate height, holes, bumpiness, and line clears. The action is
    # chosen to move toward the best placement: rotate first, then move horizontally, then hard drop.
    if env.line_clear_animation is not None or env.game_over or env.game_won:
        return [0, 0, 0]
    
    def check_collision(grid, shape, x, y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = x + c
                    grid_y = y + r
                    if grid_x < 0 or grid_x >= env.GRID_WIDTH or grid_y >= env.GRID_HEIGHT:
                        return True
                    if grid_y >= 0 and grid[grid_y][grid_x] > 0:
                        return True
        return False

    def evaluate_grid(grid):
        aggregate_height = 0
        heights = []
        for col in range(env.GRID_WIDTH):
            found = False
            for r in range(env.GRID_HEIGHT):
                if grid[r][col] > 0:
                    aggregate_height += (env.GRID_HEIGHT - r)
                    heights.append(env.GRID_HEIGHT - r)
                    found = True
                    break
            if not found:
                heights.append(0)
        
        holes = 0
        for col in range(env.GRID_WIDTH):
            found_block = False
            for r in range(env.GRID_HEIGHT):
                if grid[r][col] > 0:
                    found_block = True
                else:
                    if found_block:
                        holes += 1
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return -0.51 * aggregate_height - 0.76 * holes - 0.36 * bumpiness

    initial_shape = env.SHAPES[env.current_piece['shape_idx']]
    unique_rotations = set()
    current = initial_shape
    for _ in range(4):
        current_tup = tuple(tuple(row) for row in current)
        if current_tup in unique_rotations:
            break
        unique_rotations.add(current_tup)
        current = list(zip(*current[::-1]))
    unique_rotations = [list(list(row) for row in rot) for rot in unique_rotations]

    best_score = -10**9
    best_rotation = None
    best_x = None
    binary_grid = [[1 if cell > 0 else 0 for cell in row] for row in env.grid]

    for rotation in unique_rotations:
        width = len(rotation[0])
        for x in range(env.GRID_WIDTH - width + 1):
            temp_grid = [row[:] for row in binary_grid]
            y = 0
            while not check_collision(temp_grid, rotation, x, y + 1):
                y += 1
            for r, row_piece in enumerate(rotation):
                for c, cell in enumerate(row_piece):
                    if cell and 0 <= y + r < env.GRID_HEIGHT and 0 <= x + c < env.GRID_WIDTH:
                        temp_grid[y + r][x + c] = 1
            
            lines_cleared = 0
            full_rows = []
            for r in range(env.GRID_HEIGHT):
                if all(temp_grid[r][c] > 0 for c in range(env.GRID_WIDTH)):
                    full_rows.append(r)
            for row in full_rows:
                del temp_grid[row]
                temp_grid.insert(0, [0] * env.GRID_WIDTH)
                lines_cleared += 1
            
            score = evaluate_grid(temp_grid) + 100 * lines_cleared
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_x = x

    current_rotation_tup = tuple(tuple(row) for row in env.current_piece['shape'])
    best_rotation_tup = tuple(tuple(row) for row in best_rotation)
    if current_rotation_tup != best_rotation_tup:
        return [1, 0, 0]
    current_x = env.current_piece['x']
    if current_x < best_x:
        return [4, 0, 0]
    elif current_x > best_x:
        return [3, 0, 0]
    else:
        return [0, 0, 1]