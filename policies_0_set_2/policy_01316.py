def policy(env):
    # This policy maximizes Tetris score by evaluating all possible piece placements using a heuristic that considers lines cleared, aggregate height, holes, and bumpiness, then selecting actions to achieve the best placement.
    if env.game_over:
        return [0, 0, 0]
    
    def evaluate_placement(env, shape, x, y, shape_idx):
        temp_grid = [[env.grid[i][j] for j in range(env.GRID_WIDTH)] for i in range(env.GRID_HEIGHT)]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    i_place = y + r
                    j_place = x + c
                    if 0 <= i_place < env.GRID_HEIGHT and 0 <= j_place < env.GRID_WIDTH:
                        temp_grid[i_place][j_place] = 1
        lines_to_clear = []
        for i in range(env.GRID_HEIGHT):
            if all(temp_grid[i][j] != 0 for j in range(env.GRID_WIDTH)):
                lines_to_clear.append(i)
        num_lines = len(lines_to_clear)
        for i in sorted(lines_to_clear, reverse=True):
            del temp_grid[i]
            temp_grid.insert(0, [0] * env.GRID_WIDTH)
        min_rows = [env.GRID_HEIGHT] * env.GRID_WIDTH
        for i in range(env.GRID_HEIGHT):
            for j in range(env.GRID_WIDTH):
                if temp_grid[i][j] != 0 and i < min_rows[j]:
                    min_rows[j] = i
        heights = [0] * env.GRID_WIDTH
        for j in range(env.GRID_WIDTH):
            if min_rows[j] < env.GRID_HEIGHT:
                heights[j] = env.GRID_HEIGHT - min_rows[j]
            else:
                heights[j] = 0
        aggregate_height = sum(heights)
        holes = 0
        for j in range(env.GRID_WIDTH):
            if min_rows[j] < env.GRID_HEIGHT:
                for i in range(min_rows[j], env.GRID_HEIGHT):
                    if temp_grid[i][j] == 0:
                        holes += 1
        bumpiness = 0
        for j in range(env.GRID_WIDTH - 1):
            bumpiness += abs(heights[j] - heights[j+1])
        score = num_lines * 100 - aggregate_height * 0.5 - holes * 1.0 - bumpiness * 0.2
        return score

    current_piece = env.current_piece
    base_shape = env.SHAPES[current_piece['shape_idx']]
    best_score = -float('inf')
    best_shape = None
    best_x = None
    shape_rot = base_shape
    for rot in range(4):
        if rot > 0:
            shape_rot = [list(row) for row in zip(*shape_rot[::-1])]
        width = len(shape_rot[0])
        for x in range(env.GRID_WIDTH - width + 1):
            y = current_piece['y']
            temp_piece = {'shape': shape_rot, 'x': x, 'y': y}
            while env._is_valid_position(temp_piece, offset=(0, 1)):
                y += 1
                temp_piece['y'] = y
            if not env._is_valid_position(temp_piece, offset=(0, 0)):
                continue
            score = evaluate_placement(env, shape_rot, x, y, current_piece['shape_idx'])
            if score > best_score:
                best_score = score
                best_shape = shape_rot
                best_x = x
    current_shape = current_piece['shape']
    current_x = current_piece['x']
    if current_shape != best_shape:
        return [1, 0, 0]
    elif current_x < best_x:
        return [4, 0, 0]
    elif current_x > best_x:
        return [3, 0, 0]
    else:
        return [0, 1, 0]