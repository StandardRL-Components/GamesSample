def policy(env):
    # Strategy: Prioritize holding I pieces for potential tetris clears, then minimize stack height by moving to the lowest column and hard dropping to reduce holes and maximize line clear opportunities.
    if hasattr(env, 'line_clear_animation') and env.line_clear_animation is not None:
        return [0, 0, 0]
    
    if env.can_hold:
        if env.held_piece is not None and env.held_piece['color_idx'] == 0:
            return [0, 0, 1]
        if env.held_piece is None and env.next_piece['color_idx'] == 0:
            return [0, 0, 1]
    
    heights = [0] * 10
    for col in range(10):
        for row in range(20):
            if env.grid[row][col] != 0:
                heights[col] = 20 - row
                break
    
    min_height = min(heights)
    target_x = next(i for i, h in enumerate(heights) if h == min_height)
    piece_width = len(env.current_piece['shape'][0])
    if target_x + piece_width > 10:
        target_x = 10 - piece_width
    
    current_x = env.current_piece['x']
    if current_x < target_x:
        return [4, 0, 0]
    elif current_x > target_x:
        return [3, 0, 0]
    else:
        return [0, 1, 0]