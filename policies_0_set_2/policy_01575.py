def policy(env):
    # Strategy: Efficiently match pairs by first flipping unknown cards, then remembering patterns.
    # When one card is face-up, target its match if known, else explore systematically.
    # Avoids flipping already matched or face-up cards, and waits during mismatch checks.
    if env.game_over or env.mismatch_check_timer > 0:
        return [0, 0, 0]
    
    n_selected = len(env.selected_cards_indices)
    current_idx = env.cursor_pos[0] * env.GRID_COLS + env.cursor_pos[1]
    
    # If one card is face-up, find its match if known
    if n_selected == 1:
        up_pattern = env.cards[env.selected_cards_indices[0]]['pattern_id']
        for idx, card in enumerate(env.cards):
            if card['state'] == 'down' and card['pattern_id'] == up_pattern:
                target_idx = idx
                break
        else:
            target_idx = None
    else:
        target_idx = None
    
    # Find first face-down card if no target
    if target_idx is None:
        for idx, card in enumerate(env.cards):
            if card['state'] == 'down':
                target_idx = idx
                break
        else:
            return [0, 0, 0]
    
    # Move to target card if not already there
    if current_idx != target_idx:
        target_row = target_idx // env.GRID_COLS
        target_col = target_idx % env.GRID_COLS
        curr_row, curr_col = env.cursor_pos
        
        # Compute wrapped differences
        diff_row = (target_row - curr_row) % env.GRID_ROWS
        if diff_row > env.GRID_ROWS // 2:
            diff_row -= env.GRID_ROWS
        diff_col = (target_col - curr_col) % env.GRID_COLS
        if diff_col > env.GRID_COLS // 2:
            diff_col -= env.GRID_COLS
        
        # Choose movement direction
        if diff_row < 0:
            return [1, 0, 0]
        elif diff_row > 0:
            return [2, 0, 0]
        elif diff_col < 0:
            return [3, 0, 0]
        elif diff_col > 0:
            return [4, 0, 0]
    
    # Flip the card if on target and able
    if n_selected < 2 and env.cards[current_idx]['state'] == 'down':
        return [0, 1, 0]
    
    return [0, 0, 0]