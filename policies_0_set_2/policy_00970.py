def policy(env):
    # Strategy: Use full knowledge of symbols to always reveal matching pairs efficiently. 
    # Move to the match of the revealed tile or to the first hidden tile if no revealed tile. 
    # Press space when on target to maximize matches and minimize mismatches.
    GRID_SIZE = env.GRID_SIZE
    mismatch_timer = env.mismatch_info['timer']
    revealed_tiles = env.revealed_tiles
    grid_states = env.grid_states
    grid_symbols = env.grid_symbols
    cursor_r, cursor_c = env.cursor_pos
    can_reveal = mismatch_timer == 0
    
    target_r, target_c = None, None
    if len(revealed_tiles) > 0:
        r1, c1 = revealed_tiles[0]
        s1 = grid_symbols[r1, c1]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid_states[r, c] == 0 and grid_symbols[r, c] == s1 and (r, c) != (r1, c1):
                    target_r, target_c = r, c
                    break
            if target_r is not None:
                break
        if target_r is None:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid_states[r, c] == 0:
                        target_r, target_c = r, c
                        break
                if target_r is not None:
                    break
    else:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid_states[r, c] == 0:
                    target_r, target_c = r, c
                    break
            if target_r is not None:
                break

    if target_r is None:
        return [0, 0, 0]

    on_target = (cursor_r == target_r) and (cursor_c == target_c)
    
    if on_target and can_reveal and grid_states[target_r, target_c] == 0 and len(revealed_tiles) < 2:
        a1 = 1
    else:
        a1 = 0

    if not on_target:
        dr = (target_r - cursor_r) % GRID_SIZE
        dc = (target_c - cursor_c) % GRID_SIZE
        if dc != 0:
            if dc <= 2:
                a0 = 4
            else:
                a0 = 3
        else:
            if dr != 0:
                if dr <= 2:
                    a0 = 2
                else:
                    a0 = 1
            else:
                a0 = 0
    else:
        a0 = 0

    a2 = 0
    return [a0, a1, a2]