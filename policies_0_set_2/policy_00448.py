def policy(env):
    """
    Optimizes Tetris gameplay by evaluating all possible placements for the current piece.
    Selects the move that minimizes grid height, maximizes line clears, and minimizes holes.
    Uses hard drop for immediate placement once optimal position is reached.
    """
    if env.game_over or (env.line_clear_animation and env.line_clear_animation['timer'] > 0):
        return [0, 0, 0]
    
    current_piece = env.current_piece
    if not current_piece:
        return [0, 0, 0]
    
    piece_type = current_piece['type']
    num_rotations = len(env.PIECES[piece_type])
    best_score = -10**9
    best_rotation = current_piece['rotation']
    best_x = current_piece['x']
    
    for rotation in range(num_rotations):
        for x in range(env.GRID_WIDTH):
            temp_piece = {
                'type': piece_type,
                'rotation': rotation,
                'x': x,
                'y': 0
            }
            
            if env._check_collision(temp_piece, 0, 0):
                continue
                
            drop_y = 0
            while not env._check_collision(temp_piece, 0, drop_y + 1):
                drop_y += 1
            temp_piece['y'] += drop_y
            
            temp_grid = env.grid.copy()
            shape = env.PIECES[piece_type][rotation]
            for bx, by in shape:
                gx = temp_piece['x'] + bx
                gy = temp_piece['y'] + by
                if 0 <= gy < env.GRID_HEIGHT and 0 <= gx < env.GRID_WIDTH:
                    temp_grid[gy, gx] = piece_type + 1
            
            lines_cleared = 0
            for y in range(env.GRID_HEIGHT):
                if np.all(temp_grid[y, :] > 0):
                    lines_cleared += 1
            
            aggregate_height = 0
            holes = 0
            heights = [0] * env.GRID_WIDTH
            for col in range(env.GRID_WIDTH):
                found = False
                for row in range(env.GRID_HEIGHT):
                    if temp_grid[row, col] > 0:
                        heights[col] = env.GRID_HEIGHT - row
                        aggregate_height += heights[col]
                        found = True
                        break
                if not found:
                    heights[col] = 0
                
                for row in range(env.GRID_HEIGHT):
                    if temp_grid[row, col] == 0 and any(temp_grid[r, col] > 0 for r in range(row)):
                        holes += 1
            
            bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(env.GRID_WIDTH-1))
            
            score = 100 * lines_cleared - 0.5 * aggregate_height - 0.8 * holes - 0.3 * bumpiness
            
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_x = x
    
    current_rotation = current_piece['rotation']
    current_x = current_piece['x']
    
    if current_rotation != best_rotation:
        diff = (best_rotation - current_rotation) % num_rotations
        return [1 if diff <= num_rotations//2 else 2, 0, 0]
    
    if current_x < best_x:
        return [4, 0, 0]
    elif current_x > best_x:
        return [3, 0, 0]
    
    return [0, 0, 1]