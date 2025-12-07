def policy(env):
    # Strategy: Efficiently find hidden words by moving to word endpoints and submitting selections.
    # Prioritize closest unfound words to minimize movement time. Use edge-triggered actions to avoid no-ops.
    current_pos = env.cursor_pos
    selection_start = env.selection_start_pos
    
    if env.game_over:
        return [0, 0, 0]
    
    unfound_words = [word for word in env.target_words_info if word not in env.found_words]
    if not unfound_words:
        return [0, 0, 0]
    
    if selection_start is None:
        best_dist = float('inf')
        target_pos = None
        for word in unfound_words:
            info = env.target_words_info[word]
            for pos in [info['start'], info['end']]:
                dist = abs(pos[0] - current_pos[0]) + abs(pos[1] - current_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    target_pos = pos
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == 0 and dy == 0:
            a1 = 0 if env.last_space_held else 1
            return [0, a1, 0]
        elif abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
        return [a0, 0, 0]
    else:
        target_word = None
        target_end = None
        for word in unfound_words:
            info = env.target_words_info[word]
            if tuple(selection_start) == info['start']:
                target_word = word
                target_end = info['end']
                break
            elif tuple(selection_start) == info['end']:
                target_word = word
                target_end = info['start']
                break
        
        if target_word is None:
            if tuple(selection_start) == tuple(current_pos):
                a2 = 0 if env.last_shift_held else 1
                return [0, 0, a2]
            dx = selection_start[0] - current_pos[0]
            dy = selection_start[1] - current_pos[1]
        else:
            dx = target_end[0] - current_pos[0]
            dy = target_end[1] - current_pos[1]
        
        if dx == 0 and dy == 0:
            a2 = 0 if env.last_shift_held else 1
            return [0, 0, a2]
        elif abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
        return [a0, 0, 0]