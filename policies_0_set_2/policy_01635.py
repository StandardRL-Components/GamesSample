def policy(env):
    # Strategy: Prioritize completing words for immediate rewards. For each unfound word, check if current selection
    # is a valid prefix. If so, continue along the word's path. Otherwise, move towards the start of a new word.
    # Avoid diagonal words due to movement constraints and focus on horizontal/vertical words for reliable selection.
    
    # Get current state
    current_pos = tuple(env.cursor_pos)
    selection = env.selection_path
    unfound_words = [w for w in env.words_to_find if w not in env.found_words]
    
    # If currently selecting, try to continue the word
    if selection:
        current_word = ''.join(env.grid[y][x] for x, y in selection)
        for word in unfound_words:
            if current_word == word[:len(selection)] or current_word == word[::-1][:len(selection)]:
                # Find the full path for this word
                coords = sorted(env.word_solutions[word])
                candidate = ''.join(env.grid[y][x] for x, y in coords)
                path = coords if candidate == word else list(reversed(coords))
                
                # Check if current selection matches the path start
                if selection == path[:len(selection)]:
                    next_idx = len(selection)
                    if next_idx < len(path):
                        next_cell = path[next_idx]
                        dx = next_cell[0] - current_pos[0]
                        dy = next_cell[1] - current_pos[1]
                        if dx == -1: return [3, 0, 0]
                        if dx == 1: return [4, 0, 0]
                        if dy == -1: return [1, 0, 0]
                        if dy == 1: return [2, 0, 0]
                    else:
                        return [0, 1, 0]  # Submit completed word
        return [0, 0, 1]  # Clear invalid selection

    # Find a new word to start
    for word in unfound_words:
        coords = sorted(env.word_solutions[word])
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        # Skip diagonal words (cannot select with horizontal/vertical moves)
        if not (all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys)):
            continue
            
        candidate = ''.join(env.grid[y][x] for x, y in coords)
        path = coords if candidate == word else list(reversed(coords))
        start_cell = path[0]
        
        # Move toward start cell
        dx = start_cell[0] - current_pos[0]
        dy = start_cell[1] - current_pos[1]
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        if dy != 0:
            return [2 if dy > 0 else 1, 0, 0]
        # Already at start cell, begin selection
        return [0, 0, 0]  # No move needed; selection will start on next step

    return [0, 0, 0]  # Default no-op if no words found