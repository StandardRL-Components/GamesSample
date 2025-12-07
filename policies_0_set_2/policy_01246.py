def policy(env):
    # Strategy: Prioritize swaps that create immediate matches. If no match is found, move toward the best potential swap.
    # Evaluate adjacent swaps for the selected gem (if any) or for all gems (if none selected) to maximize matches and score.
    board = env.board
    selected = env.selected_gem_pos
    cursor_r, cursor_c = env.cursor_pos[1], env.cursor_pos[0]
    
    def is_match(brd, r, c):
        if brd[r, c] == -1:
            return False
        gem = brd[r, c]
        # Check horizontal
        left = 0
        col = c - 1
        while col >= 0 and brd[r, col] == gem:
            left += 1
            col -= 1
        right = 0
        col = c + 1
        while col < env.BOARD_WIDTH and brd[r, col] == gem:
            right += 1
            col += 1
        if 1 + left + right >= 3:
            return True
        # Check vertical
        up = 0
        row = r - 1
        while row >= 0 and brd[row, c] == gem:
            up += 1
            row -= 1
        down = 0
        row = r + 1
        while row < env.BOARD_HEIGHT and brd[row, c] == gem:
            down += 1
            row += 1
        return 1 + up + down >= 3

    if selected is not None:
        sel_r, sel_c = selected[1], selected[0]
        adjacents = []
        if sel_r > 0:
            adjacents.append((sel_r-1, sel_c, 1))  # up
        if sel_r < env.BOARD_HEIGHT-1:
            adjacents.append((sel_r+1, sel_c, 2))  # down
        if sel_c > 0:
            adjacents.append((sel_r, sel_c-1, 3))  # left
        if sel_c < env.BOARD_WIDTH-1:
            adjacents.append((sel_r, sel_c+1, 4))  # right
        
        for adj_r, adj_c, move_dir in adjacents:
            board_copy = board.copy()
            board_copy[sel_r, sel_c], board_copy[adj_r, adj_c] = board_copy[adj_r, adj_c], board_copy[sel_r, sel_c]
            if is_match(board_copy, sel_r, sel_c) or is_match(board_copy, adj_r, adj_c):
                if cursor_r == adj_r and cursor_c == adj_c:
                    return [0, 1, 0]  # Press space to swap
                else:
                    return [move_dir, 0, 0]  # Move toward adjacent gem
        return [0, 0, 1]  # Cancel selection if no valid swap

    else:
        best_swap = None
        for r in range(env.BOARD_HEIGHT):
            for c in range(env.BOARD_WIDTH):
                if c < env.BOARD_WIDTH-1:
                    board_copy = board.copy()
                    board_copy[r, c], board_copy[r, c+1] = board_copy[r, c+1], board_copy[r, c]
                    if is_match(board_copy, r, c) or is_match(board_copy, r, c+1):
                        best_swap = (r, c, 4)  # right move
                        break
                if r < env.BOARD_HEIGHT-1:
                    board_copy = board.copy()
                    board_copy[r, c], board_copy[r+1, c] = board_copy[r+1, c], board_copy[r, c]
                    if is_match(board_copy, r, c) or is_match(board_copy, r+1, c):
                        best_swap = (r, c, 2)  # down move
                        break
            if best_swap is not None:
                break
        
        if best_swap is not None:
            t_r, t_c, move_dir = best_swap
            if cursor_r == t_r and cursor_c == t_c:
                return [0, 1, 0]  # Select gem
            else:
                if cursor_r < t_r:
                    return [2, 0, 0]  # Move down
                elif cursor_r > t_r:
                    return [1, 0, 0]  # Move up
                elif cursor_c < t_c:
                    return [4, 0, 0]  # Move right
                else:
                    return [3, 0, 0]  # Move left
        return [0, 0, 0]  # No-op if no swap found