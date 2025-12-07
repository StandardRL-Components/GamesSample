def policy(env):
    # Strategy: Prioritize forming valid words from adjacent letters to clear blocks and maximize score.
    # Use a small set of common words for validation, extend selections with high-value letters,
    # and clear long invalid paths to avoid wasting moves.
    COMMON_WORDS = ("be", "in", "on", "at", "we", "he", "it", "to", "do", "go", "so", "my", "up", "by", "or", "of", "as", "is", "an", "if", "the", "and", "for", "not", "but", "you", "all", "any", "can", "had", "has", "him", "her", "one", "out", "see", "now", "are", "his", "how", "its", "let", "put", "say", "she", "too", "use", "way", "yes")
    if env.current_word.lower() in COMMON_WORDS:
        return [0, 1, 0]
    if len(env.current_word) > 5:
        return [0, 0, 1]
    x, y = env.cursor_pos
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    possible_moves = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.GRID_COLS and 0 <= ny < env.GRID_ROWS and env.grid[ny][nx] is not None:
            if env.selection_path and (nx, ny) in env.selection_path:
                continue
            letter = env.grid[ny][nx]['letter']
            value = env.LETTER_VALUES[letter]
            possible_moves.append((value, dx, dy))
    if possible_moves:
        possible_moves.sort(key=lambda x: x[0], reverse=True)
        best_value, best_dx, best_dy = possible_moves[0]
        move_action = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}[(best_dx, best_dy)]
        return [move_action, 0, 0]
    else:
        if env.selection_path:
            return [0, 0, 1]
        else:
            return [4, 0, 0]