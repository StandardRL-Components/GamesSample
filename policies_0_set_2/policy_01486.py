def policy(env):
    # Strategy: Always match pairs optimally by leveraging known grid state to avoid mistakes.
    # Since we know the entire grid (env.grid), we can always select the matching tile for the currently revealed one (if any).
    # Otherwise, we reveal the next unrevealed unmatched tile in a fixed order (row-major) to systematically uncover pairs.
    # This minimizes incorrect guesses (limited to 3) and maximizes score by ensuring every reveal leads to a match.

    def get_movement_action(current, target, grid_dim):
        r0, c0 = current
        r1, c1 = target
        dr = (r1 - r0) % grid_dim
        if dr > grid_dim // 2:
            dr -= grid_dim
        dc = (c1 - c0) % grid_dim
        if dc > grid_dim // 2:
            dc -= grid_dim
        if dc != 0:
            return 4 if dc > 0 else 3
        else:
            return 2 if dr > 0 else 1

    if env.animation_timer > 0:
        return [0, 0, 0]
    
    current = env.cursor_pos
    if env.first_selection is not None:
        n = env.grid[env.first_selection[0], env.first_selection[1]]
        target = None
        for i in range(env.GRID_DIM):
            for j in range(env.GRID_DIM):
                if (i, j) != env.first_selection and env.grid[i, j] == n and not env.matched_mask[i, j]:
                    target = (i, j)
                    break
            if target is not None:
                break
        if target is None:
            return [0, 0, 0]
        if current[0] == target[0] and current[1] == target[1]:
            return [0, 1, 0]
        else:
            move_action = get_movement_action(current, target, env.GRID_DIM)
            return [move_action, 0, 0]
    else:
        target = None
        for i in range(env.GRID_DIM):
            for j in range(env.GRID_DIM):
                if not env.matched_mask[i, j] and not env.revealed_mask[i, j]:
                    target = (i, j)
                    break
            if target is not None:
                break
        if target is None:
            return [0, 0, 0]
        if current[0] == target[0] and current[1] == target[1]:
            return [0, 1, 0]
        else:
            move_action = get_movement_action(current, target, env.GRID_DIM)
            return [move_action, 0, 0]