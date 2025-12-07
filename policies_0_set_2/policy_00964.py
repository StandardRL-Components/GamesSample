def policy(env):
    # Strategy: Efficiently find words by scanning grid for uncollected words, prioritizing those with endpoints near cursor.
    # Move to word start, then trace to end while holding space to maximize reward from successful finds.
    if env.game_over:
        return [0, 0, 0]
    
    if env.selection_start is not None:
        # Currently selecting - find target word and move to its end point
        start = env.selection_start
        unfound = sorted(set(env.WORD_LIST) - env.found_words)
        target_end = None
        for word in unfound:
            if word not in env.word_locations:
                continue
            coords = env.word_locations[word]
            if start == coords[0]:
                target_end = coords[-1]
                break
            elif start == coords[-1]:
                target_end = coords[0]
                break
        
        if target_end is None:
            return [0, 0, 1]  # Cancel invalid selection
        
        cx, cy = env.cursor_pos
        tx, ty = target_end
        if (cx, cy) == (tx, ty):
            return [0, 0, 0]  # Release space at target
        if cx < tx:
            return [4, 1, 0]  # Move right
        if cx > tx:
            return [3, 1, 0]  # Move left
        if cy < ty:
            return [2, 1, 0]  # Move down
        return [1, 1, 0]  # Move up
    
    # Not selecting - move to start of nearest uncollected word
    unfound = [w for w in sorted(set(env.WORD_LIST) - env.found_words) if w in env.word_locations]
    if not unfound:
        return [0, 0, 0]
    
    # Find closest word endpoint to cursor
    cx, cy = env.cursor_pos
    best_dist = float('inf')
    target_start = None
    for word in unfound:
        coords = env.word_locations[word]
        for endpoint in [coords[0], coords[-1]]:
            dist = abs(cx - endpoint[0]) + abs(cy - endpoint[1])
            if dist < best_dist:
                best_dist = dist
                target_start = endpoint
    
    if target_start is None:
        return [0, 0, 0]
    
    tx, ty = target_start
    if (cx, cy) == (tx, ty):
        return [0, 1, 0]  # Start selection
    if cx < tx:
        return [4, 0, 0]  # Move right
    if cx > tx:
        return [3, 0, 0]  # Move left
    if cy < ty:
        return [2, 0, 0]  # Move down
    return [1, 0, 0]  # Move up