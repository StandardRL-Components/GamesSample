def policy(env):
    # This policy uses a heuristic to maximize Tetris score by minimizing grid height and holes while prioritizing line clears.
    # It evaluates all possible moves (rotations and translations) for the current piece, simulates hard drops, and selects the action that leads to the best grid state.
    # The grid state is scored based on aggregate height, holes, bumpiness, and immediate lines cleared (using known Tetris AI weights).
    if env.game_over or env.current_block is None:
        return [0, 1, 0]
    
    def check_collision(x, y, shape, grid):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    if x + j < 0 or x + j >= len(grid[0]) or y + i >= len(grid):
                        return True
                    if y + i >= 0 and grid[y + i][x + j] != 0:
                        return True
        return False

    def evaluate_grid(grid, lines_cleared):
        n_cols = len(grid[0])
        n_rows = len(grid)
        heights = [0] * n_cols
        for x in range(n_cols):
            for y in range(n_rows):
                if grid[y][x] != 0:
                    heights[x] = n_rows - y
                    break
        aggregate_height = sum(heights)
        holes = 0
        for x in range(n_cols):
            found_block = False
            for y in range(n_rows):
                if grid[y][x] != 0:
                    found_block = True
                else:
                    if found_block:
                        holes += 1
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(n_cols - 1))
        return -0.51 * aggregate_height + 0.76 * lines_cleared - 0.36 * holes - 0.18 * bumpiness

    candidate_actions = [0, 1, 2, 3, 4]
    best_score = float('-inf')
    best_a0 = 0

    for a0 in candidate_actions:
        grid_copy = [row[:] for row in env.grid]
        block_copy = env.current_block.copy()
        block_copy['shape'] = [list(row) for row in block_copy['shape']]
        valid = True

        if a0 == 1:
            original_shape = block_copy['shape']
            new_shape = list(zip(*original_shape[::-1]))
            attempts = [(0, 0), (1, 0), (-1, 0)]
            rotated = False
            for dx, dy in attempts:
                if not check_collision(block_copy['x'] + dx, block_copy['y'] + dy, new_shape, grid_copy):
                    block_copy['x'] += dx
                    block_copy['y'] += dy
                    block_copy['shape'] = new_shape
                    rotated = True
                    break
            if not rotated:
                continue
        elif a0 == 2:
            original_shape = block_copy['shape']
            new_shape = list(zip(*original_shape))[::-1]
            attempts = [(0, 0), (1, 0), (-1, 0)]
            rotated = False
            for dx, dy in attempts:
                if not check_collision(block_copy['x'] + dx, block_copy['y'] + dy, new_shape, grid_copy):
                    block_copy['x'] += dx
                    block_copy['y'] += dy
                    block_copy['shape'] = new_shape
                    rotated = True
                    break
            if not rotated:
                continue
        elif a0 == 3:
            if check_collision(block_copy['x'] - 1, block_copy['y'], block_copy['shape'], grid_copy):
                continue
            block_copy['x'] -= 1
        elif a0 == 4:
            if check_collision(block_copy['x'] + 1, block_copy['y'], block_copy['shape'], grid_copy):
                continue
            block_copy['x'] += 1

        while not check_collision(block_copy['x'], block_copy['y'] + 1, block_copy['shape'], grid_copy):
            block_copy['y'] += 1

        for y, row in enumerate(block_copy['shape']):
            for x, cell in enumerate(row):
                if cell:
                    gy = block_copy['y'] + y
                    gx = block_copy['x'] + x
                    if 0 <= gy < len(grid_copy) and 0 <= gx < len(grid_copy[0]):
                        grid_copy[gy][gx] = block_copy['id'] + 1

        lines_cleared = 0
        lines_to_clear = []
        for y, row in enumerate(grid_copy):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(y)
        for y in lines_to_clear:
            grid_copy.pop(y)
            grid_copy.insert(0, [0] * len(grid_copy[0]))
            lines_cleared += 1

        score = evaluate_grid(grid_copy, lines_cleared)
        if score > best_score:
            best_score = score
            best_a0 = a0

    return [best_a0, 1, 0]