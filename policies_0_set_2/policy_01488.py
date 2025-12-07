def policy(env):
    """
    Strategy: Prioritize placing crystals in the beam path to redirect light toward unlit targets.
    Moves cursor to nearest empty beam cell for placement, or explores unlit targets if no beam cells available.
    Uses Manhattan distance for efficient navigation and avoids unnecessary placements.
    """
    if env.game_over:
        return [0, 0, 0]
    
    cursor = tuple(env.cursor_pos)
    grid = env.grid
    remaining = env.remaining_placement_crystals
    lit = env.lit_crystals
    targets = env.target_crystals
    unlit = [t for t in targets if t not in lit]
    
    # Get all beam cells from light path segments
    beam_cells = set()
    for start, end in env.light_path_segments:
        x1, y1 = start
        x2, y2 = end
        if x1 == x2:  # vertical
            for y in range(min(y1, y2), max(y1, y2) + 1):
                beam_cells.add((x1, y))
        else:  # horizontal
            for x in range(min(x1, x2), max(x1, x2) + 1):
                beam_cells.add((x, y1))
    
    # Find empty beam cells
    empty_beam = [c for c in beam_cells if grid[c] == 0]
    
    if empty_beam and remaining > 0:
        # Find closest empty beam cell
        target_cell = min(empty_beam, key=lambda c: abs(c[0]-cursor[0]) + abs(c[1]-cursor[1]))
        if cursor == target_cell:
            return [0, 1, 0]
        # Move toward target cell
        dx = target_cell[0] - cursor[0]
        dy = target_cell[1] - cursor[1]
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    
    # No beam cells available, explore unlit targets
    if unlit:
        target = min(unlit, key=lambda t: abs(t[0]-cursor[0]) + abs(t[1]-cursor[1]))
        dx = target[0] - cursor[0]
        dy = target[1] - cursor[1]
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    
    return [0, 0, 0]