def policy(env):
    # This policy maximizes reward by prioritizing moves that create matches (immediate rewards) and then sets up future matches.
    # It first checks if a crystal is selected, then evaluates adjacent swaps for matches. If none found, it deselects.
    # If no crystal is selected, it moves the cursor to the best cell (based on potential match count and distance) and selects it.
    # The strategy ensures efficient use of moves by always seeking the highest immediate reward and avoiding invalid moves.
    
    def find_matches(board):
        matches = set()
        h, w = board.shape
        # Horizontal matches
        for r in range(h):
            for c in range(w - 2):
                if board[r, c] != 0 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    matches.update((r, c+i) for i in range(3))
        # Vertical matches
        for c in range(w):
            for r in range(h - 2):
                if board[r, c] != 0 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    matches.update((r+i, c) for i in range(3))
        return matches
    
    if env.game_over:
        return [0, 0, 0]
    
    if env.selected_pos is not None:
        r0, c0 = env.selected_pos
        best_dir = 0
        best_matches = 0
        for dir, (dr, dc) in enumerate([(0,0), (-1,0), (1,0), (0,-1), (0,1)][1:], 1):
            r1, c1 = r0 + dr, c0 + dc
            if 0 <= r1 < env.GRID_HEIGHT and 0 <= c1 < env.GRID_WIDTH:
                board_copy = env.board.copy()
                board_copy[r0, c0], board_copy[r1, c1] = board_copy[r1, c1], board_copy[r0, c0]
                matches = find_matches(board_copy)
                if len(matches) > best_matches:
                    best_matches = len(matches)
                    best_dir = dir
        if best_matches > 0:
            return [best_dir, 0, 0]
        else:
            return [0, 1, 0]
    
    else:
        best_score = -1
        best_pos = None
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                if env.board[r, c] == 0:
                    continue
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < env.GRID_HEIGHT and 0 <= c2 < env.GRID_WIDTH and env.board[r2, c2] != 0:
                        board_copy = env.board.copy()
                        board_copy[r, c], board_copy[r2, c2] = board_copy[r2, c2], board_copy[r, c]
                        matches = find_matches(board_copy)
                        score = len(matches)
                        if score > best_score:
                            best_score = score
                            best_pos = (r, c)
        if best_pos is None:
            return [0, 0, 0]
        target_r, target_c = best_pos
        curr_r, curr_c = env.cursor_pos
        if curr_r == target_r and curr_c == target_c:
            return [0, 1, 0]
        if curr_c != target_c:
            return [4 if target_c > curr_c else 3, 0, 0]
        else:
            return [2 if target_r > curr_r else 1, 0, 0]