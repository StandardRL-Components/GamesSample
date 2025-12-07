def policy(env):
    # Strategy: Maximize score by painting incorrect pixels with correct color, then moving to next incorrect pixel.
    # Avoids penalties by only painting when color matches target and pixel is incorrect.
    cursor_x, cursor_y = env.cursor_pos
    target_color = env.target_image[cursor_y, cursor_x]
    current_color = env.canvas[cursor_y, cursor_x]
    selected_color = env.selected_color_idx
    
    if current_color != target_color:
        if selected_color == target_color:
            a1 = 1 if not env.prev_space_held else 0
            return [0, a1, 0]
        else:
            a2 = 1 if not env.prev_shift_held else 0
            return [0, 0, a2]
    else:
        wrong_pixels = []
        for y in range(env.CANVAS_GRID_SIZE):
            for x in range(env.CANVAS_GRID_SIZE):
                if env.canvas[y, x] != env.target_image[y, x]:
                    wrong_pixels.append((x, y))
        
        if not wrong_pixels:
            return [0, 0, 0]
            
        best_dist = float('inf')
        best_pixel = wrong_pixels[0]
        for (x, y) in wrong_pixels:
            dx = min(abs(x - cursor_x), env.CANVAS_GRID_SIZE - abs(x - cursor_x))
            dy = min(abs(y - cursor_y), env.CANVAS_GRID_SIZE - abs(y - cursor_y))
            dist = dx + dy
            if dist < best_dist:
                best_dist = dist
                best_pixel = (x, y)
                
        tx, ty = best_pixel
        dx = (tx - cursor_x + env.CANVAS_GRID_SIZE//2) % env.CANVAS_GRID_SIZE - env.CANVAS_GRID_SIZE//2
        dy = (ty - cursor_y + env.CANVAS_GRID_SIZE//2) % env.CANVAS_GRID_SIZE - env.CANVAS_GRID_SIZE//2
        
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
            
        return [a0, 0, 0]