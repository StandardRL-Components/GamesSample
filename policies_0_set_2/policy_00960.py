def policy(env):
    # Strategy: Prioritize immediate matches by scanning for valid swaps (right/down) in row-major order.
    # If a tile is selected, move to the first valid adjacent swap partner and complete the swap.
    # Otherwise, select the first tile that can form a match when swapped with an adjacent tile.
    # Movement uses wrap-aware direction choices to minimize Manhattan distance.
    if env.game_over:
        return [0, 0, 0]
    GRID_SIZE = env.GRID_SIZE
    board = env.board
    cursor_r, cursor_c = env.cursor_pos
    selected = env.selected_tile

    def find_match(temp_board):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE - 2):
                if temp_board[r][c] != 0 and temp_board[r][c] == temp_board[r][c+1] == temp_board[r][c+2]:
                    return True
        for c in range(GRID_SIZE):
            for r in range(GRID_SIZE - 2):
                if temp_board[r][c] != 0 and temp_board[r][c] == temp_board[r+1][c] == temp_board[r+2][c]:
                    return True
        return False

    def get_move(cur_r, cur_c, target_r, target_c):
        dr = (target_r - cur_r) % GRID_SIZE
        if dr > GRID_SIZE // 2:
            dr -= GRID_SIZE
        dc = (target_c - cur_c) % GRID_SIZE
        if dc > GRID_SIZE // 2:
            dc -= GRID_SIZE
        if dr == 0 and dc == 0:
            return None
        if abs(dr) > abs(dc):
            return 1 if dr < 0 else 2
        else:
            return 3 if dc < 0 else 4

    if selected is not None:
        sel_r, sel_c = selected
        for dr, dc in [(-1,0), (0,1), (1,0), (0,-1)]:
            r2, c2 = sel_r + dr, sel_c + dc
            if 0 <= r2 < GRID_SIZE and 0 <= c2 < GRID_SIZE:
                temp_board = [list(row) for row in board]
                temp_board[sel_r][sel_c], temp_board[r2][c2] = temp_board[r2][c2], temp_board[sel_r][sel_c]
                if find_match(temp_board):
                    move = get_move(cursor_r, cursor_c, r2, c2)
                    if move is None:
                        return [0, 1, 0]
                    else:
                        return [move, 0, 0]
        return [0, 0, 1]

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            for dr, dc in [(0,1), (1,0)]:
                r2, c2 = r + dr, c + dc
                if r2 < GRID_SIZE and c2 < GRID_SIZE:
                    temp_board = [list(row) for row in board]
                    temp_board[r][c], temp_board[r2][c2] = temp_board[r2][c2], temp_board[r][c]
                    if find_match(temp_board):
                        move = get_move(cursor_r, cursor_c, r, c)
                        if move is None:
                            return [0, 1, 0]
                        else:
                            return [move, 0, 0]
    return [0, 0, 0]