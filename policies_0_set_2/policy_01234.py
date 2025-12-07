def policy(env):
    # This policy uses a greedy approach to maximize immediate and future rewards in Tetris.
    # It evaluates all possible placements of the current piece by simulating drops for each rotation and horizontal position.
    # The best placement is chosen based on lines cleared, aggregate height, and holes to minimize stack height and maximize clears.
    # Actions are then selected to rotate and move the piece to the target placement, followed by a hard drop when aligned.
    if env.game_over or env.clear_animation_timer > 0:
        return [0, 0, 0]
    
    def evaluate_board(board):
        lines_cleared = 0
        new_board = []
        for row in board:
            if all(cell != 0 for cell in row):
                lines_cleared += 1
            else:
                new_board.append(row)
        for _ in range(lines_cleared):
            new_board.insert(0, [0] * len(board[0]))
        
        aggregate_height = 0
        holes = 0
        num_cols = len(board[0])
        num_rows = len(new_board)
        for col in range(num_cols):
            top_row = None
            for row in range(num_rows):
                if new_board[row][col] != 0:
                    top_row = row
                    break
            if top_row is None:
                continue
            height = num_rows - top_row
            aggregate_height += height
            for row in range(top_row + 1, num_rows):
                if new_board[row][col] == 0:
                    holes += 1
        return lines_cleared * 100 - aggregate_height * 1 - holes * 10

    board = [[env.board[y, x] for x in range(env.GRID_WIDTH)] for y in range(env.GRID_HEIGHT)]
    current_piece = env.current_piece
    shape_idx = current_piece['shape_idx']
    num_rotations = len(env.PIECE_SHAPES[shape_idx])
    
    best_score = -10**9
    best_rotation = current_piece['rotation']
    best_x = current_piece['x']
    
    for rotation in range(num_rotations):
        for x in range(env.GRID_WIDTH):
            piece_copy = {
                'shape_idx': shape_idx,
                'rotation': rotation,
                'x': x,
                'y': 0,
                'color_idx': current_piece['color_idx']
            }
            if not env._is_valid_position(piece_copy):
                continue
                
            y = 0
            while env._is_valid_position(piece_copy, (0, y + 1)):
                y += 1
            piece_copy['y'] = y
            
            if not env._is_valid_position(piece_copy):
                continue
                
            temp_board = [row[:] for row in board]
            coords = env._get_piece_coords(piece_copy)
            for (x_coord, y_coord) in coords:
                if 0 <= y_coord < env.GRID_HEIGHT and 0 <= x_coord < env.GRID_WIDTH:
                    temp_board[y_coord][x_coord] = 1
                    
            score = evaluate_board(temp_board)
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_x = x
                
    current_rotation = current_piece['rotation']
    current_x = current_piece['x']
    
    if current_rotation != best_rotation:
        cw_steps = (best_rotation - current_rotation) % num_rotations
        ccw_steps = (current_rotation - best_rotation) % num_rotations
        if cw_steps <= ccw_steps:
            return [1, 0, 0]
        else:
            return [0, 0, 1]
            
    if current_x != best_x:
        if current_x < best_x:
            return [4, 0, 0]
        else:
            return [3, 0, 0]
            
    return [0, 1, 0]