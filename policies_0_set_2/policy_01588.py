def policy(env):
    # Strategy: Avoid bombs, prioritize slicing fruits in columns with highest adjusted score (points + bonus for adjacent bombs).
    # Move to the safest and most rewarding column, only slice when current column has non-negative adjusted score.
    current_col = env.cursor_pos
    bomb_cols = [b['col'] for b in env.bombs]
    safe_cols = [col for col in range(env.GRID_WIDTH) if col not in bomb_cols]
    
    best_col = current_col
    best_score = -10.0
    for col in safe_cols:
        fruits_in_col = [f for f in env.fruits if f['col'] == col]
        if fruits_in_col:
            base_score = sum(f['points'] for f in fruits_in_col)
            adjacent_bomb = any(abs(b_col - col) == 1 for b_col in bomb_cols)
            adjusted_score = base_score + (5 if adjacent_bomb else -2)
        else:
            adjusted_score = -10.0
        if adjusted_score > best_score or (adjusted_score == best_score and abs(col - current_col) < abs(best_col - current_col)):
            best_score = adjusted_score
            best_col = col
    
    if not safe_cols:
        best_col = current_col
        best_score = -10.0
    
    movement = 0
    if current_col < best_col:
        movement = 4
    elif current_col > best_col:
        movement = 3
        
    slice_action = 1 if (current_col == best_col and best_score >= 0) else 0
    return [movement, slice_action, 0]