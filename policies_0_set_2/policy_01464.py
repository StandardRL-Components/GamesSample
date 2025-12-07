def policy(env):
    # Strategy: Prioritize placing crystals in the current light path to redirect beams towards unlit targets.
    # Evaluate candidate positions in the light beam that can refract towards targets, maximizing immediate rewards.
    # Move cursor efficiently to optimal positions, cycle crystal type if needed, and place when aligned.

    if env.game_over:
        return [0, 0, 0]

    unlit_targets = [pos for i, pos in enumerate(env.targets) if not env.targets_lit_status[i]]
    if not unlit_targets or env.crystals_remaining <= 0:
        return [0, 0, 0]

    light_cells = set()
    for segment in env.light_path_segments:
        for cell in segment:
            if 0 <= cell[0] < env.GRID_W and 0 <= cell[1] < env.GRID_H:
                light_cells.add(cell)

    best_score = -float('inf')
    best_action = [0, 0, 0]
    cursor_x, cursor_y = env.cursor_pos

    for cell in light_cells:
        if cell in env.crystals_placed or cell in env.walls or cell in env.targets or cell == env.light_source['pos']:
            continue

        for crystal_type in range(3):
            score = 0
            for target in unlit_targets:
                dx = target[0] - cell[0]
                dy = target[1] - cell[1]
                dist = abs(dx) + abs(dy)
                score -= dist
            if score > best_score:
                best_score = score
                target_x, target_y = cell
                move_x = 4 if target_x > cursor_x else 3 if target_x < cursor_x else 0
                move_y = 2 if target_y > cursor_y else 1 if target_y < cursor_y else 0
                move = move_x if move_x != 0 else move_y
                cycle = 1 if crystal_type != env.selected_crystal_type else 0
                place = 1 if cell == env.cursor_pos and cycle == 0 else 0
                best_action = [move, place, cycle]

    return best_action