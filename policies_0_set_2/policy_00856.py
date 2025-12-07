def policy(env):
    """
    This policy maximizes score by prioritizing immediate matches from pushes, then moving to crystals that can create matches.
    It avoids costly shuffles and no-ops by simulating push outcomes and moving efficiently toward high-value targets.
    """
    def simulate_push(r, c, direction):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[direction]
        line = []
        r_cur, c_cur = r, c
        while 0 <= r_cur < env.BOARD_SIZE and 0 <= c_cur < env.BOARD_SIZE and board[r_cur, c_cur] != 0:
            line.append((r_cur, c_cur))
            r_cur += dr
            c_cur += dc
        if not (0 <= r_cur < env.BOARD_SIZE and 0 <= c_cur < env.BOARD_SIZE) or board[r_cur, c_cur] != 0:
            return 0
        board_copy = board.copy()
        for (r_p, c_p) in reversed(line):
            board_copy[r_p + dr, c_p + dc] = board_copy[r_p, c_p]
        board_copy[r, c] = 0
        matches = 0
        for r_i in range(env.BOARD_SIZE):
            for c_i in range(env.BOARD_SIZE - 2):
                if board_copy[r_i, c_i] != 0 and board_copy[r_i, c_i] == board_copy[r_i, c_i+1] == board_copy[r_i, c_i+2]:
                    matches += 3
        for c_i in range(env.BOARD_SIZE):
            for r_i in range(env.BOARD_SIZE - 2):
                if board_copy[r_i, c_i] != 0 and board_copy[r_i, c_i] == board_copy[r_i+1, c_i] == board_copy[r_i+2, c_i]:
                    matches += 3
        return matches

    board = env.board
    r, c = env.cursor_pos
    last_dir = env.last_move_direction

    if last_dir is not None and last_dir != 0 and board[r, c] != 0:
        match_count = simulate_push(r, c, last_dir)
        if match_count > 0:
            return [0, 1, 0]

    best_score = -1
    best_action = [0, 0, 0]
    for r_i in range(env.BOARD_SIZE):
        for c_i in range(env.BOARD_SIZE):
            if board[r_i, c_i] == 0:
                continue
            for direction in [1, 2, 3, 4]:
                score = simulate_push(r_i, c_i, direction)
                if score > best_score:
                    best_score = score
                    dr = r_i - r
                    dc = c_i - c
                    if dr != 0 or dc != 0:
                        move_dir = 2 if dr > 0 else 1 if dr < 0 else 4 if dc > 0 else 3
                        best_action = [move_dir, 0, 0]
                    else:
                        best_action = [direction, 0, 0] if direction != last_dir else [0, 1, 0]

    if best_score > 0:
        return best_action

    for move_dir in [1, 2, 3, 4]:
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[move_dir]
        r_new, c_new = r + dr, c + dc
        if 0 <= r_new < env.BOARD_SIZE and 0 <= c_new < env.BOARD_SIZE and board[r_new, c_new] != 0:
            return [move_dir, 0, 0]

    return [0, 0, 1]