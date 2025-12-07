def policy(env):
    # Strategy: Use a deterministic Mastermind solver that generates consistent guesses based on previous feedback.
    # For each row, compute the set of codes consistent with all previous feedback, then choose the lexicographically first code.
    # Fill the current row with this code, then submit. This maximizes the expected reward by efficiently narrowing down possibilities.

    if env.game_over:
        return [0, 0, 0]
    
    def evaluate_guess(secret, guess):
        black = white = 0
        s_copy, g_copy = list(secret), list(guess)
        for i in range(len(guess)):
            if g_copy[i] == s_copy[i]:
                black += 1
                s_copy[i] = g_copy[i] = -1
        for i in range(len(guess)):
            if g_copy[i] != -1 and g_copy[i] in s_copy:
                white += 1
                s_copy[s_copy.index(g_copy[i])] = -1
        return black, white

    n_colors, n_slots = env.NUM_COLORS, env.BOARD_COLS
    current_guess = env.guesses[env.current_row]
    
    if all(peg != -1 for peg in current_guess):
        return [0, 0, 1]
    
    all_codes = []
    for a in range(n_colors):
        for b in range(n_colors):
            for c in range(n_colors):
                for d in range(n_colors):
                    all_codes.append([a, b, c, d])
    
    S = []
    for code in all_codes:
        valid = True
        for i in range(env.current_row):
            if env.feedback[i][0] != -1:
                black, white = evaluate_guess(code, env.guesses[i])
                if (black, white) != env.feedback[i]:
                    valid = False
                    break
        if valid:
            S.append(code)
    
    S.sort()
    target_code = S[0] if S else [0] * n_slots
    
    empty_col = next(i for i, peg in enumerate(current_guess) if peg == -1)
    target_color = target_code[empty_col]
    
    if env.cursor_x != empty_col:
        return [4 if env.cursor_x < empty_col else 3, 0, 0]
    
    if env.selected_color_idx != target_color:
        clockwise = (target_color - env.selected_color_idx) % n_colors
        counterclockwise = (env.selected_color_idx - target_color) % n_colors
        return [2 if clockwise <= counterclockwise else 1, 0, 0]
    
    return [0, 1, 0]