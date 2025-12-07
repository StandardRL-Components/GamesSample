def policy(env):
    # Strategy: Maximize score by clearing lines and minimizing holes/height. 
    # For current piece, evaluate all valid x positions (without rotation) and choose the one that minimizes holes and aggregate height while maximizing potential lines cleared.
    # Move horizontally to target x, then hard drop when aligned. Use soft drop only when already aligned.
    if env.game_over:
        return [0, 0, 0]
    
    current_piece = env.current_piece
    current_x = current_piece["x"]
    shape = current_piece["shape"]
    grid = env.grid
    
    best_x = current_x
    best_score = -float('inf')
    
    for x in range(env.GRID_WIDTH):
        test_piece = {"shape": shape, "x": x, "y": 0}
        if not env._is_valid_position(test_piece):
            continue
            
        y = 0
        while True:
            test_piece["y"] = y + 1
            if not env._is_valid_position(test_piece):
                break
            y += 1
        test_piece["y"] = y
        
        holes = 0
        aggregate_height = 0
        temp_grid = grid.copy()
        
        for dx, dy in shape:
            px = x + dx
            py = y + dy
            if 0 <= px < env.GRID_WIDTH and 0 <= py < env.GRID_HEIGHT:
                temp_grid[px, py] = 1
                
        for col in range(env.GRID_WIDTH):
            col_height = 0
            found_block = False
            for row in range(env.GRID_HEIGHT - 1, -1, -1):
                if temp_grid[col, row] != 0:
                    col_height = env.GRID_HEIGHT - row
                    found_block = True
                    break
            aggregate_height += col_height
            if found_block:
                for row in range(env.GRID_HEIGHT - col_height, env.GRID_HEIGHT):
                    if temp_grid[col, row] == 0:
                        holes += 1
        
        lines_cleared = 0
        for row in range(env.GRID_HEIGHT):
            if all(temp_grid[col, row] != 0 for col in range(env.GRID_WIDTH)):
                lines_cleared += 1
                
        score = -holes - aggregate_height + 10 * lines_cleared
        if score > best_score:
            best_score = score
            best_x = x
            
    if current_x < best_x:
        return [4, 0, 0]
    elif current_x > best_x:
        return [3, 0, 0]
    else:
        if env.last_action[1] == 0:
            return [0, 1, 0]
        else:
            return [2, 0, 0]