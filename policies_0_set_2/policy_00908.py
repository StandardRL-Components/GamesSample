def policy(env):
    """
    Strategy: This is a memory matching game where the goal is to find all pairs before time runs out.
    We systematically scan the grid row by row, flipping hidden cards to reveal symbols.
    When we have one card revealed, we immediately try to find its match by checking adjacent cells first.
    We avoid flipping during mismatch cooldown and prioritize matching known pairs when possible.
    """
    # Check if we're in mismatch cooldown - do nothing during this period
    if hasattr(env, 'mismatch_timer') and env.mismatch_timer > 0:
        return [0, 0, 0]
    
    r, c = env.cursor_pos
    grid = env.grid
    
    # If we have a first selection already revealed, find its match
    if env.first_selection is not None:
        r1, c1 = env.first_selection
        target_symbol = grid[r1][c1]['symbol']
        
        # Check adjacent cells first for efficiency
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = (r + dr) % env.GRID_ROWS, (c + dc) % env.GRID_COLS
            if grid[nr][nc]['state'] == 'hidden' and grid[nr][nc]['symbol'] == target_symbol:
                # Move toward matching card
                if dr == -1: return [1, 0, 0]
                if dr == 1: return [2, 0, 0]
                if dc == -1: return [3, 0, 0]
                if dc == 1: return [4, 0, 0]
        
        # If match not adjacent, move systematically through grid
        for nr in range(env.GRID_ROWS):
            for nc in range(env.GRID_COLS):
                if grid[nr][nc]['state'] == 'hidden' and grid[nr][nc]['symbol'] == target_symbol:
                    # Move toward target position
                    if nr < r: return [1, 0, 0]
                    if nr > r: return [2, 0, 0]
                    if nc < c: return [3, 0, 0]
                    if nc > c: return [4, 0, 0]
        return [0, 0, 0]
    
    # If current card is hidden and no selection made, flip it
    if grid[r][c]['state'] == 'hidden':
        return [0, 1, 0]
    
    # Otherwise find next hidden card using systematic grid search
    for nr in range(env.GRID_ROWS):
        for nc in range(env.GRID_COLS):
            if grid[nr][nc]['state'] == 'hidden':
                # Move toward next hidden card
                if nr < r: return [1, 0, 0]
                if nr > r: return [2, 0, 0]
                if nc < c: return [3, 0, 0]
                if nc > c: return [4, 0, 0]
    
    # If no hidden cards found (game should be over), do nothing
    return [0, 0, 0]