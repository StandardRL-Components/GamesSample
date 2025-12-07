def policy(env):
    # Strategy: Draw a continuous downward slope from the rider's position to the finish line.
    # This maximizes speed via gravity while avoiding penalties for short lines or inaction.
    # If drawing, extend line 100px right and 50px down from start; else move cursor to rider.
    if env.game_over:
        return [0, 0, 0]
    if env.current_line_start is None:
        target_x, target_y = env.rider_pos.x, env.rider_pos.y
        dx = target_x - env.drawing_cursor_pos.x
        dy = target_y - env.drawing_cursor_pos.y
        if abs(dx) < 8 and abs(dy) < 8:
            return [0, 0, 1]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    else:
        target_x = env.current_line_start.x + 100
        target_y = env.current_line_start.y + 50
        dx = target_x - env.drawing_cursor_pos.x
        dy = target_y - env.drawing_cursor_pos.y
        if abs(dx) < 8 and abs(dy) < 8:
            return [0, 1, 0]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]