def policy(env):
    # Strategy: Prioritize connecting closest valid pairs first to minimize moves and avoid crossings.
    # In state 1 (no selection), choose the nearest uncompleted dot from the closest valid pair.
    # In state 2 (selected), move directly to the matching dot if valid, else cancel and retry.
    if env.selected_dot_index is not None:
        selected_dot = env.dots[env.selected_dot_index]
        color = selected_dot['color']
        dot_indices = env.color_groups[color]['dots']
        other_idx = dot_indices[0] if dot_indices[0] != env.selected_dot_index else dot_indices[1]
        other_dot = env.dots[other_idx]
        if env._validate_connection(env.selected_dot_index, other_idx):
            dx = other_dot['pos'][0] - env.cursor_pos[0]
            dy = other_dot['pos'][1] - env.cursor_pos[1]
            if dx == 0 and dy == 0:
                return [0, 1, 0]
            if abs(dx) > abs(dy):
                a0 = 4 if dx > 0 else 3
            else:
                a0 = 2 if dy > 0 else 1
            return [a0, 0, 0]
        else:
            return [0, 0, 1]
    else:
        uncompleted = [c for c in env.color_groups if not env._is_color_group_complete(c)]
        if not uncompleted:
            return [0, 0, 0]
        best_group = None
        best_dist = float('inf')
        for color in uncompleted:
            dots = env.color_groups[color]['dots']
            dot1 = env.dots[dots[0]]
            dot2 = env.dots[dots[1]]
            dist = abs(dot1['pos'][0]-dot2['pos'][0]) + abs(dot1['pos'][1]-dot2['pos'][1])
            if dist < best_dist and env._validate_connection(dots[0], dots[1]):
                best_dist = dist
                best_group = color
        if best_group is None:
            best_group = uncompleted[0]
        dots = env.color_groups[best_group]['dots']
        dot1 = env.dots[dots[0]]
        dot2 = env.dots[dots[1]]
        d1 = abs(dot1['pos'][0]-env.cursor_pos[0]) + abs(dot1['pos'][1]-env.cursor_pos[1])
        d2 = abs(dot2['pos'][0]-env.cursor_pos[0]) + abs(dot2['pos'][1]-env.cursor_pos[1])
        target = dot1 if d1 <= d2 else dot2
        dx = target['pos'][0] - env.cursor_pos[0]
        dy = target['pos'][1] - env.cursor_pos[1]
        if dx == 0 and dy == 0:
            return [0, 1, 0]
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
        return [a0, 0, 0]