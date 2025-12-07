def policy(env):
    # This policy uses a heuristic to maximize line clears and minimize grid height by evaluating
    # all possible placements for the current piece. It selects the placement that minimizes
    # aggregate height and holes while maximizing potential line clears, then issues actions to
    # achieve that placement. Hard drop is used once aligned to maximize efficiency.
    if env.game_over or env.line_clear_anim_timer > 0:
        return [0, 0, 0]
    
    current_piece = env.current_piece
    shape_key = current_piece['shape_key']
    n_rotations = len(env.SHAPES[shape_key])
    best_score = -10**9
    best_rotation = current_piece['rotation']
    best_x = current_piece['x']
    
    grid_list = [[env.grid[r, c] for c in range(env.GRID_WIDTH)] for r in range(env.GRID_HEIGHT)]
    
    for rot in range(n_rotations):
        for x in range(env.GRID_WIDTH):
            temp_piece = {'shape_key': shape_key, 'rotation': rot, 'x': x, 'y': 0}
            if env._check_collision(temp_piece, 0, 0):
                continue
                
            drop_y = 0
            while not env._check_collision(temp_piece, 0, drop_y + 1):
                drop_y += 1
                
            sim_grid = [row[:] for row in grid_list]
            color_index = list(env.SHAPE_COLORS.values()).index(current_piece['color']) + 1
            shape = env.SHAPES[shape_key][rot]
            for dx, dy in shape:
                sim_x, sim_y = x + dx, drop_y + dy
                if 0 <= sim_x < env.GRID_WIDTH and 0 <= sim_y < env.GRID_HEIGHT:
                    sim_grid[sim_y][sim_x] = color_index
                    
            lines_cleared = 0
            new_grid = []
            for row in sim_grid:
                if all(cell != 0 for cell in row):
                    lines_cleared += 1
                else:
                    new_grid.append(row)
            for _ in range(lines_cleared):
                new_grid.insert(0, [0] * env.GRID_WIDTH)
                
            heights = [0] * env.GRID_WIDTH
            for c in range(env.GRID_WIDTH):
                for r in range(env.GRID_HEIGHT):
                    if new_grid[r][c] != 0:
                        heights[c] = env.GRID_HEIGHT - r
                        break
            aggregate_height = sum(heights)
            
            holes = 0
            for c in range(env.GRID_WIDTH):
                found = False
                for r in range(env.GRID_HEIGHT):
                    if new_grid[r][c] != 0:
                        found = True
                    elif found:
                        holes += 1
            bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(env.GRID_WIDTH-1))
            
            score = lines_cleared * 100 - aggregate_height * 0.5 - holes * 10 - bumpiness * 0.5
            if score > best_score:
                best_score = score
                best_rotation = rot
                best_x = x
                
    current_rot = current_piece['rotation']
    current_x = current_piece['x']
    action = [0, 0, 0]
    
    if current_rot != best_rotation:
        diff = (best_rotation - current_rot) % n_rotations
        if diff <= n_rotations // 2:
            action[0] = 1
        else:
            action[2] = 1
    else:
        if current_x < best_x:
            action[0] = 4
        elif current_x > best_x:
            action[0] = 3
        else:
            action[1] = 1
            
    return action