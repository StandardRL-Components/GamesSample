def policy(env):
    # Strategy: Prioritize completing valid words for immediate rewards, then extend current word along valid prefixes.
    # Use BFS to find nearest valid word completion or prefix extension, avoiding dead ends and revisits.
    from collections import deque

    # Check if current word is valid and not found
    if env.current_word in env.WORD_SET and env.current_word not in env.completed_words:
        return [0, 1, 0]  # Submit current word

    r, c = env.worm_head
    visited = set(env.worm_path)
    queue = deque([(r, c, [], 0)])
    best_dir = 0
    best_score = -float('inf')
    directions = [(1,0,'down'), (-1,0,'up'), (0,1,'right'), (0,-1,'left')]
    dir_map = {'up':1, 'down':2, 'left':3, 'right':4}

    # BFS for nearest valid extension
    while queue:
        cr, cc, path, dist = queue.popleft()
        current_str = env.current_word + ''.join(env.grid[r][c] for (r,c) in path)
        
        # Score based on word validity and distance
        if current_str in env.WORD_SET and current_str not in env.completed_words:
            score = 1000 - dist  # Prioritize closer completions
        elif current_str in env.PREFIX_SET:
            score = 100 - dist  # Prefer closer prefixes
        else:
            score = -dist  # Penalize invalid paths

        if score > best_score:
            best_score = score
            best_dir = dir_map[path[0][2]] if path else 0

        # Expand neighbors
        for dr, dc, name in directions:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < env.GRID_ROWS and 0 <= nc < env.GRID_COLS and 
                (nr, nc) not in visited and len(path) < 5):  # Limit search depth
                queue.append((nr, nc, path + [(nr, nc, name)], dist + 1))
                visited.add((nr, nc))

    return [best_dir, 0, 0]  # Move toward best found path