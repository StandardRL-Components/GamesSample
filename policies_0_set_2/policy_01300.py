def policy(env):
    # Strategy: Prioritize correcting incorrect pixels by moving to the nearest mismatch and painting with the target color.
    # If current color is wrong, cycle colors to match target. Avoid using exhausted colors to prevent game over.
    target = env.target_image
    player = env.player_canvas
    color_counts = env.color_counts
    selected_color = env.selected_color_index
    cursor = env.cursor_pos
    bg_index = env.BG_COLOR_INDEX

    # If current color is exhausted, cycle to next available color
    if color_counts[selected_color] <= 0:
        return [0, 0, 1]

    # Find all incorrect non-BG pixels with available correct color
    incorrect_pixels = []
    for y in range(env.CANVAS_H):
        for x in range(env.CANVAS_W):
            t_color = target[y, x]
            if t_color == bg_index:
                continue
            if player[y, x] != t_color and color_counts[t_color] > 0:
                incorrect_pixels.append((x, y))

    if not incorrect_pixels:
        return [0, 0, 0]

    # Find closest incorrect pixel using Manhattan distance with wrapping
    best_dist = float('inf')
    best_pixel = None
    for (x, y) in incorrect_pixels:
        dx = min(abs(x - cursor[0]), env.CANVAS_W - abs(x - cursor[0]))
        dy = min(abs(y - cursor[1]), env.CANVAS_H - abs(y - cursor[1]))
        dist = dx + dy
        if dist < best_dist:
            best_dist = dist
            best_pixel = (x, y)

    tx, ty = best_pixel
    t_color = target[ty, tx]

    # If at target pixel
    if cursor[0] == tx and cursor[1] == ty:
        if selected_color == t_color:
            return [0, 1, 0]  # Paint
        else:
            return [0, 0, 1]  # Cycle color

    # Move towards target pixel
    dx = (tx - cursor[0]) % env.CANVAS_W
    if dx > env.CANVAS_W // 2:
        dx -= env.CANVAS_W
    dy = (ty - cursor[1]) % env.CANVAS_H
    if dy > env.CANVAS_H // 2:
        dy -= env.CANVAS_H

    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]