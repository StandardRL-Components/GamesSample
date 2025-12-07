def policy(env):
    # Strategy: Maximize immediate matches by evaluating all possible adjacent swaps on the 3x3 grid.
    # Prioritize swaps that create the most matches (reward), breaking ties by position order.
    # During animations, return no-op. If selected tile isn't optimal, deselect and recompute.
    
    if env.animation_state != 'IDLE':
        return [0, 0, 0]
    
    def count_matches(grid, r1, c1, r2, c2):
        temp = grid.copy()
        temp[r1, c1], temp[r2, c2] = temp[r2, c2], temp[r1, c1]
        matched = set()
        for r in range(3):
            if temp[r, 0] == temp[r, 1] == temp[r, 2] != -1:
                matched.update([(r, 0), (r, 1), (r, 2)])
        for c in range(3):
            if temp[0, c] == temp[1, c] == temp[2, c] != -1:
                matched.update([(0, c), (1, c), (2, c)])
        return len(matched)
    
    best_score = -1
    best_swap = None
    grid = env.grid
    for r in range(3):
        for c in range(3):
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 3 and 0 <= c2 < 3:
                    score = count_matches(grid, r, c, r2, c2)
                    if score > best_score:
                        best_score = score
                        best_swap = (r, c, r2, c2)
    
    if best_swap is None:
        return [0, 0, 0]
    
    r1, c1, r2, c2 = best_swap
    if env.selected_tile is None:
        cr, cc = env.cursor
        if (cr, cc) == (r1, c1):
            return [0, 1, 0]
        else:
            dr = r1 - cr
            dc = c1 - cc
            if dc != 0:
                move = 3 if dc < 0 else 4
            else:
                move = 1 if dr < 0 else 2
            return [move, 0, 0]
    else:
        if env.selected_tile == (r1, c1):
            dr = r2 - r1
            dc = c2 - c1
            move = [0, 1, 2, 3, 4][(dr, dc).index(1) + 1] if abs(dr) == 1 else [0, 3, 4, 1, 2][(dr, dc).index(1) + 1]
            return [move, 1, 0]
        else:
            return [0, 0, 1]