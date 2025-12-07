def policy(env):
    # This policy maximizes Tetris score by evaluating all possible placements of the current piece.
    # For each rotation and column, it simulates a hard drop and scores the resulting board based on:
    #   - Lines cleared (prioritizing immediate rewards)
    #   - Aggregate height (lower is better)
    #   - Bumpiness (smoother boards are better)
    #   - Holes (fewer are better)
    # It then chooses the action that moves toward the highest-scoring placement.

    if env.line_clear_animation is not None or env.game_over or env.victory:
        return [0, 0, 0]

    current_block = env.current_block
    shape_index = current_block['shape_index']
    current_rotation = current_block['rotation']
    current_col = current_block['pos'][1]
    num_rotations = len(env.SHAPES[shape_index])
    
    best_score = float('-inf')
    best_rotation = current_rotation
    best_column = current_col
    best_lines_cleared = 0

    for rotation in range(num_rotations):
        for column in range(env.PLAYFIELD_WIDTH):
            if env._check_collision(rotation, [0, column]):
                continue
                
            temp_pos = [0, column]
            while not env._check_collision(rotation, temp_pos):
                temp_pos[0] += 1
            temp_pos[0] -= 1

            simulated_playfield = [list(row) for row in env.playfield]
            coords = env._get_shape_coords(shape_index, rotation, temp_pos)
            for r, c in coords:
                if 0 <= r < env.PLAYFIELD_HEIGHT and 0 <= c < env.PLAYFIELD_WIDTH:
                    simulated_playfield[r][c] = 1

            lines_cleared = 0
            for r in range(env.PLAYFIELD_HEIGHT):
                if all(simulated_playfield[r][c] != 0 for c in range(env.PLAYFIELD_WIDTH)):
                    lines_cleared += 1
                    for r_above in range(r, 0, -1):
                        simulated_playfield[r_above] = simulated_playfield[r_above - 1][:]
                    simulated_playfield[0] = [0] * env.PLAYFIELD_WIDTH

            aggregate_height = 0
            heights = [0] * env.PLAYFIELD_WIDTH
            for c in range(env.PLAYFIELD_WIDTH):
                for r in range(env.PLAYFIELD_HEIGHT):
                    if simulated_playfield[r][c] != 0:
                        heights[c] = env.PLAYFIELD_HEIGHT - r
                        break
                aggregate_height += heights[c]

            bumpiness = 0
            for i in range(env.PLAYFIELD_WIDTH - 1):
                bumpiness += abs(heights[i] - heights[i + 1])

            holes = 0
            for c in range(env.PLAYFIELD_WIDTH):
                found_block = False
                for r in range(env.PLAYFIELD_HEIGHT):
                    if simulated_playfield[r][c] != 0:
                        found_block = True
                    elif found_block:
                        holes += 1

            score = lines_cleared * 100 - aggregate_height * 0.5 - bumpiness * 0.3 - holes * 1.0
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_column = column
                best_lines_cleared = lines_cleared

    rotations_needed = (best_rotation - current_rotation) % num_rotations
    if rotations_needed != 0:
        return [1, 0, 0]
    if current_col < best_column:
        return [4, 0, 0]
    if current_col > best_column:
        return [3, 0, 0]
    return [0, 1, 0] if best_lines_cleared > 0 else [2, 0, 0]