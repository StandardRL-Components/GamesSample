def policy(env):
    # Strategy: Systematically find words by moving to start, selecting, then moving to end and submitting.
    # This maximizes reward by efficiently completing words without unnecessary actions or penalties.
    unfound_words = [word for i, word in enumerate(env.target_words) if not env.found_words_mask[i] and word in env.word_metadata]
    if not unfound_words:
        return [0, 0, 0]
    
    word = unfound_words[0]
    path = env.word_metadata[word]['path']
    start = path[0]
    end = path[-1]
    
    if env.selection_start is None:
        cx, cy = env.cursor_pos
        sx, sy = start
        if cx == sx and cy == sy:
            return [0, 1, 0]
        else:
            dx = (sx - cx + env.GRID_SIZE) % env.GRID_SIZE
            if dx > env.GRID_SIZE // 2:
                dx -= env.GRID_SIZE
            dy = (sy - cy + env.GRID_SIZE) % env.GRID_SIZE
            if dy > env.GRID_SIZE // 2:
                dy -= env.GRID_SIZE
                
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 0, 0]
            else:
                return [2 if dy > 0 else 1, 0, 0]
                
    elif env.selection_end is None:
        cx, cy = env.cursor_pos
        ex, ey = end
        if cx == ex and cy == ey:
            return [0, 0, 1]
        else:
            dx = (ex - cx + env.GRID_SIZE) % env.GRID_SIZE
            if dx > env.GRID_SIZE // 2:
                dx -= env.GRID_SIZE
            dy = (ey - cy + env.GRID_SIZE) % env.GRID_SIZE
            if dy > env.GRID_SIZE // 2:
                dy -= env.GRID_SIZE
                
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 0, 0]
            else:
                return [2 if dy > 0 else 1, 0, 0]
    else:
        return [0, 0, 1]