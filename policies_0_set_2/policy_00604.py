def policy(env):
    # This policy maximizes reward by efficiently connecting dots of the same color without crossing lines.
    # It prioritizes completing connections when a dot is selected, and selects dots with valid connections otherwise.
    # Movement uses toroidal distance calculations to navigate the wrapping grid efficiently.
    if env.game_over:
        return [0, 0, 0]
    
    def toroidal_manhattan(p1, p2):
        dx = abs(p1[0] - p2[0])
        dx = min(dx, env.GRID_COLS - dx)
        dy = abs(p1[1] - p2[1])
        dy = min(dy, env.GRID_ROWS - dy)
        return dx + dy

    def get_movement_action(current, target):
        dx = (target[0] - current[0]) % env.GRID_COLS
        if dx > env.GRID_COLS // 2:
            dx -= env.GRID_COLS
        dy = (target[1] - current[1]) % env.GRID_ROWS
        if dy > env.GRID_ROWS // 2:
            dy -= env.GRID_ROWS
        
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3
        else:
            return 2 if dy > 0 else 1

    cursor = env.cursor_pos
    if env.selected_dot is not None:
        selected_pos, color_idx = env.selected_dot
        valid_targets = []
        for dot_pos in env.dots_by_color[color_idx]:
            if dot_pos != selected_pos and env._is_path_valid(selected_pos, dot_pos):
                valid_targets.append(dot_pos)
        
        if valid_targets:
            best_target = min(valid_targets, key=lambda p: toroidal_manhattan(cursor, p))
            if cursor == best_target:
                return [0, 1, 0]
            move_action = get_movement_action(cursor, best_target)
            return [move_action, 0, 0]
        else:
            if cursor == selected_pos:
                return [0, 1, 0]
            move_action = get_movement_action(cursor, selected_pos)
            return [move_action, 0, 0]
    else:
        candidate_dots = []
        for color_idx, dot_list in enumerate(env.dots_by_color):
            if len(dot_list) < 2:
                continue
            for dot_pos in dot_list:
                for other_pos in dot_list:
                    if dot_pos != other_pos and env._is_path_valid(dot_pos, other_pos):
                        candidate_dots.append(dot_pos)
                        break
        if not candidate_dots:
            return [0, 0, 0]
        best_dot = min(candidate_dots, key=lambda p: toroidal_manhattan(cursor, p))
        if cursor == best_dot:
            return [0, 1, 0]
        move_action = get_movement_action(cursor, best_dot)
        return [move_action, 0, 0]