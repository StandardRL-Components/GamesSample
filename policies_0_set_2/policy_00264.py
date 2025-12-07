def policy(env):
    # Strategy: Use env state to know card values and revealed status. When no card is selected, 
    # move to and select an unrevealed card with a known match. When one card is selected, 
    # move to and select its matching card. This efficiently completes pairs for maximum reward.
    def has_match(r, c):
        val = env.grid[r, c]
        for ri in range(env.GRID_ROWS):
            for ci in range(env.GRID_COLS):
                if (ri != r or ci != c) and not env.revealed[ri, ci] and env.grid[ri, ci] == val:
                    return True
        return False

    def get_move(r_cur, c_cur, r_tar, c_tar):
        dr = (r_tar - r_cur) % env.GRID_ROWS
        if dr > env.GRID_ROWS // 2:
            dr -= env.GRID_ROWS
        dc = (c_tar - c_cur) % env.GRID_COLS
        if dc > env.GRID_COLS // 2:
            dc -= env.GRID_COLS
        if abs(dr) > abs(dc):
            return 2 if dr > 0 else 1
        else:
            return 4 if dc > 0 else 3

    eff_sel = [] if env.is_mismatch_state else env.selected_cards
    n_sel = len(eff_sel)
    r_cur, c_cur = env.cursor_pos

    if n_sel == 0:
        target = None
        for r in range(env.GRID_ROWS):
            for c in range(env.GRID_COLS):
                if not env.revealed[r, c] and has_match(r, c):
                    target = (r, c)
                    break
            if target is not None:
                break
        if target is None:
            return [0, 0, 0]
        r_tar, c_tar = target
        if r_cur == r_tar and c_cur == c_tar:
            return [0, 1, 0]
        else:
            move = get_move(r_cur, c_cur, r_tar, c_tar)
            return [move, 0, 0]
    elif n_sel == 1:
        r1, c1 = eff_sel[0]
        val = env.grid[r1, c1]
        target = None
        for r in range(env.GRID_ROWS):
            for c in range(env.GRID_COLS):
                if not env.revealed[r, c] and (r != r1 or c != c1) and env.grid[r, c] == val:
                    target = (r, c)
                    break
            if target is not None:
                break
        if target is None:
            return [0, 0, 0]
        r_tar, c_tar = target
        if r_cur == r_tar and c_cur == c_tar:
            return [0, 1, 0]
        else:
            move = get_move(r_cur, c_cur, r_tar, c_tar)
            return [move, 0, 0]
    else:
        return [0, 0, 0]