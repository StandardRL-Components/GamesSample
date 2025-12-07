def policy(env):
    # Strategy: Systematically sweep the grid in row-major order to reveal all safe squares.
    # This minimizes mine risk by avoiding random moves and ensures complete coverage.
    # We prioritize revealing hidden cells (immediate reward) and avoid already revealed/flagged cells.
    info = env._get_info()
    if env.game_over or info['win']:
        return [0, 0, 0]
    steps = info['steps']
    cursor = info['cursor_pos']
    target_index = steps % 25
    target_x = target_index % 5
    target_y = target_index // 5
    if cursor[0] == target_x and cursor[1] == target_y:
        return [0, 1, 0]
    else:
        dx = (target_x - cursor[0]) % 5
        if dx > 2:
            dx -= 5
        dy = (target_y - cursor[1]) % 5
        if dy > 2:
            dy -= 5
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]