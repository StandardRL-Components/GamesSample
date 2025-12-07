def policy(env):
    # This policy maximizes reward by prioritizing cells that complete rows/columns for bonuses, then correct individual cells.
    # It moves directly to the best candidate (considering distance and potential bonus) and sets the correct color before painting.
    cursor = env.cursor_pos
    grid_size = env.GRID_SIZE
    player_grid = env.player_grid
    target_image = env.target_image
    completed_rows = env.completed_rows
    completed_cols = env.completed_cols
    current_color = env.selected_color_idx

    best_candidate = None
    best_score = -9999
    best_distance = 9999
    for y in range(grid_size):
        for x in range(grid_size):
            if player_grid[y, x] == target_image[y, x]:
                continue
            row_correct = 0
            for i in range(grid_size):
                if player_grid[y, i] == target_image[y, i]:
                    row_correct += 1
            col_correct = 0
            for i in range(grid_size):
                if player_grid[i, x] == target_image[i, x]:
                    col_correct += 1
            new_row_correct = row_correct + 1
            new_col_correct = col_correct + 1
            row_complete = (new_row_correct == grid_size) and (y not in completed_rows)
            col_complete = (new_col_correct == grid_size) and (x not in completed_cols)
            score = 1 + 5 * (row_complete + col_complete)
            dx = min(abs(x - cursor[0]), grid_size - abs(x - cursor[0]))
            dy = min(abs(y - cursor[1]), grid_size - abs(y - cursor[1]))
            distance = dx + dy
            if score > best_score or (score == best_score and distance < best_distance):
                best_score = score
                best_distance = distance
                best_candidate = (x, y)
                
    if best_candidate is None:
        return [0, 0, 0]
        
    x, y = best_candidate
    target_color = target_image[y, x]
    if cursor[0] == x and cursor[1] == y:
        if current_color == target_color:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
            
    dx = (x - cursor[0]) % grid_size
    if dx > grid_size // 2:
        dx -= grid_size
    dy = (y - cursor[1]) % grid_size
    if dy > grid_size // 2:
        dy -= grid_size
        
    if dx != 0:
        movement = 3 if dx < 0 else 4
    else:
        movement = 1 if dy < 0 else 2
        
    space_action = 1 if current_color != target_color else 0
    return [movement, space_action, 0]