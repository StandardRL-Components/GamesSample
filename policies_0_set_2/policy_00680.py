def policy(env):
    # Strategy: Maximize score by clearing lines and minimizing stack height. Evaluate candidate placements
    # for immediate line clears, then low height and few holes. Rotate and move to best placement, then hard drop.
    if env.game_over:
        return [0, 0, 0]
    
    best_score = -10**9
    best_placement = None
    current_id = env.current_piece['id']
    current_rot = env.current_piece['rotation']
    current_x = env.current_piece['x']
    
    for rotation in range(4):
        shape = env.ROTATED_SHAPES[current_id][rotation]
        width = len(shape[0])
        for x in range(env.GRID_WIDTH - width + 1):
            y = 0
            while env._is_valid_position({'id': current_id, 'rotation': rotation, 'x': x, 'y': y}):
                y += 1
            y -= 1
            if y < 0:
                continue
                
            grid_copy = [row[:] for row in env.grid.tolist()]
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        grid_y = y + r_idx
                        grid_x = x + c_idx
                        if 0 <= grid_y < env.GRID_HEIGHT and 0 <= grid_x < env.GRID_WIDTH:
                            grid_copy[grid_y][grid_x] = current_id + 1
            
            lines_cleared = 0
            for r in range(env.GRID_HEIGHT):
                if all(grid_copy[r][c] != 0 for c in range(env.GRID_WIDTH)):
                    lines_cleared += 1
            
            heights = [0] * env.GRID_WIDTH
            for c in range(env.GRID_WIDTH):
                for r in range(env.GRID_HEIGHT):
                    if grid_copy[r][c] != 0:
                        heights[c] = env.GRID_HEIGHT - r
                        break
            
            holes = 0
            for c in range(env.GRID_WIDTH):
                found_block = False
                for r in range(env.GRID_HEIGHT):
                    if grid_copy[r][c] != 0:
                        found_block = True
                    else:
                        if found_block:
                            holes += 1
            
            score = lines_cleared * 1000 - max(heights) * 10 - holes * 50
            if score > best_score:
                best_score = score
                best_placement = (rotation, x)
    
    if best_placement is None:
        return [0, 0, 0]
    
    target_rot, target_x = best_placement
    if current_rot != target_rot:
        d_cw = (target_rot - current_rot) % 4
        d_ccw = (current_rot - target_rot) % 4
        if d_cw <= d_ccw:
            return [1, 0, 0]
        else:
            return [0, 0, 1]
    elif current_x != target_x:
        if target_x > current_x:
            return [4, 0, 0]
        else:
            return [3, 0, 0]
    else:
        return [0, 1, 0]