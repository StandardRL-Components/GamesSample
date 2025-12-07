def policy(env):
    # This policy maximizes reward by prioritizing immediate word formation using the current letter,
    # then moving to adjacent empty cells near existing letters to set up future words, and cycling the hand
    # only if a better letter for the current position is available. It avoids invalid placements and oscillations.
    grid, hand, cursor, found = env.grid, env.letter_hand, env.cursor_pos, env.found_words
    y, x = cursor
    if grid[y, x] == ' ':
        current_letter = hand[env.selected_letter_idx]
        temp_grid = grid.copy()
        temp_grid[y, x] = current_letter
        words = set()
        # Check horizontal
        start_x = x
        while start_x > 0 and temp_grid[y, start_x-1] != ' ':
            start_x -= 1
        end_x = x
        while end_x < env.GRID_SIZE-1 and temp_grid[y, end_x+1] != ' ':
            end_x += 1
        word_h = ''.join(temp_grid[y, start_x:end_x+1])
        if len(word_h) >= 3 and word_h in env.WORD_LIST and word_h not in found:
            words.add(word_h)
        # Check vertical
        start_y = y
        while start_y > 0 and temp_grid[start_y-1, x] != ' ':
            start_y -= 1
        end_y = y
        while end_y < env.GRID_SIZE-1 and temp_grid[end_y+1, x] != ' ':
            end_y += 1
        word_v = ''.join(temp_grid[start_y:end_y+1, x])
        if len(word_v) >= 3 and word_v in env.WORD_LIST and word_v not in found:
            words.add(word_v)
        if words:
            return [0, 1, 0]
        else:
            for i, letter in enumerate(hand):
                if i == env.selected_letter_idx:
                    continue
                temp_grid = grid.copy()
                temp_grid[y, x] = letter
                words = set()
                # Check horizontal
                start_x = x
                while start_x > 0 and temp_grid[y, start_x-1] != ' ':
                    start_x -= 1
                end_x = x
                while end_x < env.GRID_SIZE-1 and temp_grid[y, end_x+1] != ' ':
                    end_x += 1
                word_h = ''.join(temp_grid[y, start_x:end_x+1])
                if len(word_h) >= 3 and word_h in env.WORD_LIST and word_h not in found:
                    words.add(word_h)
                # Check vertical
                start_y = y
                while start_y > 0 and temp_grid[start_y-1, x] != ' ':
                    start_y -= 1
                end_y = y
                while end_y < env.GRID_SIZE-1 and temp_grid[end_y+1, x] != ' ':
                    end_y += 1
                word_v = ''.join(temp_grid[start_y:end_y+1, x])
                if len(word_v) >= 3 and word_v in env.WORD_LIST and word_v not in found:
                    words.add(word_v)
                if words:
                    return [0, 0, 1]
    best_score, best_move = -1, 0
    for move, (dy, dx) in enumerate([(0,0), (-1,0), (1,0), (0,-1), (0,1)], start=0):
        if move == 0:
            continue
        ny, nx = (y + dy) % env.GRID_SIZE, (x + dx) % env.GRID_SIZE
        if grid[ny, nx] != ' ':
            continue
        score = 0
        for ddy, ddx in [(-1,0), (1,0), (0,-1), (0,1)]:
            nny, nnx = (ny + ddy) % env.GRID_SIZE, (nx + ddx) % env.GRID_SIZE
            if grid[nny, nnx] != ' ':
                score += 1
        if score > best_score:
            best_score, best_move = score, move
    return [best_move, 0, 0]