def policy(env):
    # This policy uses a greedy approach to maximize immediate line clears and minimize board height.
    # It simulates all possible placements of the current piece and selects the one that minimizes
    # the number of holes and aggregate height while maximizing potential line clears.
    current_piece = env.current_piece
    piece_type = current_piece['type']
    num_rotations = len(env.PIECES[piece_type])
    best_score = float('-inf')
    best_action = [0, 0, 0]
    
    for rotation in range(num_rotations):
        for x in range(-2, env.GRID_WIDTH + 2):
            test_piece = {'type': piece_type, 'rotation': rotation, 'x': x, 'y': 0}
            if env._check_collision(0, 0, test_piece):
                continue
                
            while not env._check_collision(0, 1, test_piece):
                test_piece['y'] += 1
                
            temp_grid = env.grid.copy()
            shape = env._get_piece_shape(test_piece)
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        grid_y = test_piece['y'] + r
                        grid_x = test_piece['x'] + c
                        if 0 <= grid_y < env.GRID_HEIGHT and 0 <= grid_x < env.GRID_WIDTH:
                            temp_grid[grid_y, grid_x] = piece_type
                            
            lines_cleared = 0
            for r in range(env.GRID_HEIGHT):
                if all(temp_grid[r, c] > 0 for c in range(env.GRID_WIDTH)):
                    lines_cleared += 1
                    
            heights = [0] * env.GRID_WIDTH
            holes = 0
            for c in range(env.GRID_WIDTH):
                column_has_block = False
                for r in range(env.GRID_HEIGHT):
                    if temp_grid[r, c] > 0:
                        column_has_block = True
                        heights[c] = env.GRID_HEIGHT - r
                        break
                if column_has_block:
                    for r2 in range(r, env.GRID_HEIGHT):
                        if temp_grid[r2, c] == 0:
                            holes += 1
                            
            aggregate_height = sum(heights)
            bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(env.GRID_WIDTH-1))
            score = -0.51 * aggregate_height - 0.76 * holes - 0.36 * bumpiness + 0.18 * lines_cleared
            
            if score > best_score:
                best_score = score
                target_rotation = rotation
                target_x = x
                
    current_rotation = current_piece['rotation']
    current_x = current_piece['x']
    
    if current_rotation != target_rotation:
        return [1, 0, 0]
    elif current_x < target_x:
        return [4, 0, 0]
    elif current_x > target_x:
        return [3, 0, 0]
    else:
        return [0, 1, 0]