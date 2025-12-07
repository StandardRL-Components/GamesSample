def policy(env):
    # Strategy: Prioritize placing words to hit enemies, moving cursor to optimal positions, and cycling words when necessary.
    # Maximizes reward by focusing on immediate enemy damage and strategic placement near path/tower.
    occupied = set((w['pos'][0], w['pos'][1]) for w in env.placed_words)
    enemies = env.enemies
    if enemies:
        rounded_enemies = [(round(e['grid_pos'][0]), round(e['grid_pos'][1])) for e in enemies]
        current_word = env.available_words[env.selected_word_index]
        pattern = env.WORD_BANK[current_word]["pattern"]
        cx, cy = env.cursor_pos
        if (cx, cy) not in occupied:
            hits = sum(1 for dx, dy in pattern 
                      if (cx + dx, cy + dy) in rounded_enemies)
            if hits > 0:
                return [0, 1, 0]
        best_cell, best_hits, best_dist = None, 0, float('inf')
        for x in range(env.GRID_COLS):
            for y in range(env.GRID_ROWS):
                if (x, y) in occupied:
                    continue
                hits = sum(1 for dx, dy in pattern 
                          if (x + dx, y + dy) in rounded_enemies)
                dist = abs(x - cx) + abs(y - cy)
                if hits > best_hits or (hits == best_hits and dist < best_dist):
                    best_cell, best_hits, best_dist = (x, y), hits, dist
        if best_cell:
            dx = best_cell[0] - cx
            dy = best_cell[1] - cy
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 0, 0]
            elif dy != 0:
                return [2 if dy > 0 else 1, 0, 0]
        return [0, 0, 1]
    else:
        tx, ty = 2, 8
        dx, dy = tx - env.cursor_pos[0], ty - env.cursor_pos[1]
        if dx == 0 and dy == 0:
            return [0, 0, 0]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]