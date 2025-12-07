def policy(env):
    # Strategy: Evaluate all possible placements for the current tetromino by simulating drops at each column and rotation.
    # Choose the placement that minimizes holes, maximizes potential line clears, and maintains a flat surface.
    # Use a weighted scoring system that prioritizes line clears, minimizes height, and penalizes holes and unevenness.
    if env.current_block is None:
        return [0, 0, 0]
    
    def is_valid_placement(rot, col, row):
        shape = env.TETROMINOES[env.current_block['shape_id']][rot]
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    grid_r, grid_c = row + r, col + c
                    if grid_r >= env.PLAYFIELD_HEIGHT or grid_c < 0 or grid_c >= env.PLAYFIELD_WIDTH:
                        return False
                    if env.grid[grid_r, grid_c] != 0:
                        return False
        return True

    def simulate_drop(rot, col):
        row = env.current_block['row']
        while is_valid_placement(rot, col, row + 1):
            row += 1
        return row

    def evaluate_placement(rot, col, drop_row):
        temp_grid = [list(row) for row in env.grid]
        shape = env.TETROMINOES[env.current_block['shape_id']][rot]
        color_idx = list(env.TETROMINOES.keys()).index(env.current_block['shape_id']) + 1
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    temp_grid[drop_row + r][col + c] = color_idx

        lines_cleared = 0
        for r in range(env.PLAYFIELD_HEIGHT):
            if all(temp_grid[r][c] != 0 for c in range(env.PLAYFIELD_WIDTH)):
                lines_cleared += 1
                temp_grid.pop(r)
                temp_grid.insert(0, [0] * env.PLAYFIELD_WIDTH)

        heights = [0] * env.PLAYFIELD_WIDTH
        for c in range(env.PLAYFIELD_WIDTH):
            for r in range(env.PLAYFIELD_HEIGHT):
                if temp_grid[r][c] != 0:
                    heights[c] = env.PLAYFIELD_HEIGHT - r
                    break

        aggregate_height = sum(heights)
        holes = 0
        for c in range(env.PLAYFIELD_WIDTH):
            block_found = False
            for r in range(env.PLAYFIELD_HEIGHT):
                if temp_grid[r][c] != 0:
                    block_found = True
                elif block_found:
                    holes += 1

        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(env.PLAYFIELD_WIDTH-1))
        
        score = lines_cleared * 100 - aggregate_height * 0.5 - holes * 1.0 - bumpiness * 0.2
        return score

    best_score = -float('inf')
    best_rot = env.current_block['rotation']
    best_col = env.current_block['col']
    shape_id = env.current_block['shape_id']
    n_rotations = len(env.TETROMINOES[shape_id])

    for rot in range(n_rotations):
        shape = env.TETROMINOES[shape_id][rot]
        width = len(shape[0])
        for col in range(env.PLAYFIELD_WIDTH - width + 1):
            drop_row = simulate_drop(rot, col)
            if not is_valid_placement(rot, col, drop_row):
                continue
            score = evaluate_placement(rot, col, drop_row)
            if score > best_score:
                best_score = score
                best_rot = rot
                best_col = col

    current_rot = env.current_block['rotation']
    current_col = env.current_block['col']
    
    if env.rotate_timer == 0 and current_rot != best_rot:
        return [1, 0, 0]
    if env.move_timer == 0:
        if current_col < best_col:
            return [4, 0, 0]
        elif current_col > best_col:
            return [3, 0, 0]
    if current_rot == best_rot and current_col == best_col:
        return [0, 1, 0]
    return [0, 0, 0]