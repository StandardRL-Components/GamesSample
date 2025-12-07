def policy(env):
    # Strategy: Build a stable tower by placing large tiles centered on the base, then stack smaller tiles
    # exactly on top to minimize instability. Prioritize reaching maximum height without collapse.
    current_tile_idx = env.selected_tile_idx
    cursor_x, cursor_y = env.cursor_pos
    current_z = env.current_place_height

    # Check if target position (5,5) is already occupied at current height
    placed_at_target = False
    for tile in env.placed_tiles:
        if tile['pos'][2] == current_z:
            x, y, z = tile['pos']
            w, d = tile['type']['size']
            if x <= 5 < x + w and y <= 5 < y + d:
                placed_at_target = True
                break

    want_to_place = not placed_at_target
    a0 = 0
    a1 = 0
    a2 = 1 if current_tile_idx != 0 else 0

    if current_tile_idx == 0:
        target_x, target_y = 5, 5
        if cursor_x < target_x:
            a0 = 4
        elif cursor_x > target_x:
            a0 = 3
        elif cursor_y < target_y:
            a0 = 2
        elif cursor_y > target_y:
            a0 = 1
        else:
            if want_to_place:
                a1 = 1
    return [a0, a1, a2]