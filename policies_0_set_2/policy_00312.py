def policy(env):
    # Strategy: Maximize reward by forming the longest valid words to clear tiles efficiently.
    # Use BFS from each cell to find the longest valid word, then navigate to its start.
    # During selection, extend the current path to the best available adjacent letter.
    if not hasattr(policy, 'prefixes'):
        policy.prefixes = set()
        for word in env.WORD_LIST:
            for i in range(3, len(word) + 1):
                policy.prefixes.add(word[:i])
                
    if env.is_selecting:
        current_word = ''.join(env.grid[r][c] for r, c in env.current_path).lower()
        r, c = env.current_path[-1]
        best_dir = 0
        for dr, dc, move in [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in env.current_path and env.grid[nr][nc] != ' ':
                new_word = current_word + env.grid[nr][nc].lower()
                if new_word in policy.prefixes:
                    best_dir = move
                    break
        if best_dir != 0:
            return [best_dir, 1, 0]
        elif current_word in env.WORD_LIST:
            return [0, 0, 0]
        else:
            return [0, 0, 0]
    else:
        best_path = None
        best_len = 0
        for r in range(5):
            for c in range(5):
                if env.grid[r][c] == ' ':
                    continue
                stack = [(r, c, [(r, c)], env.grid[r][c].lower())]
                visited = {(r, c)}
                while stack:
                    cr, cc, path, word = stack.pop()
                    if word in env.WORD_LIST and len(path) > best_len:
                        best_path = path
                        best_len = len(path)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in visited and env.grid[nr][nc] != ' ':
                            new_word = word + env.grid[nr][nc].lower()
                            if new_word in policy.prefixes:
                                new_path = path + [(nr, nc)]
                                stack.append((nr, nc, new_path, new_word))
                                visited.add((nr, nc))
        if best_path is not None:
            tr, tc = best_path[0]
            r, c = env.cursor_pos
            if r < tr:
                return [2, 0, 0]
            elif r > tr:
                return [1, 0, 0]
            elif c < tc:
                return [4, 0, 0]
            elif c > tc:
                return [3, 0, 0]
            else:
                return [0, 1, 0]
        for r in range(5):
            for c in range(5):
                if env.grid[r][c] != ' ':
                    tr, tc = r, c
                    cr, cc = env.cursor_pos
                    if cr < tr:
                        return [2, 0, 0]
                    elif cr > tr:
                        return [1, 0, 0]
                    elif cc < tc:
                        return [4, 0, 0]
                    elif cc > tc:
                        return [3, 0, 0]
        return [0, 0, 0]