def policy(env):
    # Strategy: Maximize score by forming valid words efficiently. Prioritize completing existing valid words for immediate reward,
    # then extend current word if it remains a valid prefix. Use BFS to find promising word paths when no current selection.
    # Avoid invalid moves by checking adjacency and word list prefixes.
    grid = env.grid
    cursor = env.cursor_pos
    selected = env.selected_path
    current_word = env.current_word
    word_list = env.word_list

    def is_prefix(s):
        s = s.lower()
        return any(word.startswith(s) for word in word_list)

    def get_adjacent(pos):
        r, c = pos
        adjacent = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = (r + dr) % 4, (c + dc) % 4
            if (nr, nc) not in selected:
                adjacent.append((nr, nc))
        return adjacent

    if selected:
        last_pos = selected[-1]
        if current_word.lower() in word_list:
            non_adjacent = [(r, c) for r in range(4) for c in range(4) 
                           if (r, c) not in selected and not (abs(r - last_pos[0]) <= 1 and abs(c - last_pos[1]) <= 1)]
            if non_adjacent:
                target = min(non_adjacent, key=lambda p: abs(p[0]-cursor[0]) + abs(p[1]-cursor[1]))
                dr, dc = target[0]-cursor[0], target[1]-cursor[1]
                move = 2 if dr > 0 else 1 if dr < 0 else 4 if dc > 0 else 3 if dc < 0 else 0
                return [move, 1, 0]

        adjacent = get_adjacent(last_pos)
        valid_adjacent = [p for p in adjacent if is_prefix(current_word + grid[p[0]][p[1]])]
        if valid_adjacent:
            target = min(valid_adjacent, key=lambda p: abs(p[0]-cursor[0]) + abs(p[1]-cursor[1]))
            dr, dc = target[0]-cursor[0], target[1]-cursor[1]
            move = 2 if dr > 0 else 1 if dr < 0 else 4 if dc > 0 else 3 if dc < 0 else 0
            return [move, 1, 0]
        else:
            return [0, 0, 1]

    else:
        for r in range(4):
            for c in range(4):
                if is_prefix(grid[r][c]):
                    dr, dc = r-cursor[0], c-cursor[1]
                    move = 2 if dr > 0 else 1 if dr < 0 else 4 if dc > 0 else 3 if dc < 0 else 0
                    return [move, 1, 0]
        return [0, 1, 0]