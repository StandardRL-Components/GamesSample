def policy(env):
    # Strategy: Prioritize clearing groups when extra moves available, else move directly toward exit.
    # This maximizes reward by harvesting groups when possible without risking exit path, ensuring win.
    cursor = env.cursor_pos
    exit_pos = env.exit_pos
    if cursor == exit_pos:
        return [0, 0, 0]
    
    required_moves = abs(cursor[0] - exit_pos[0]) + abs(cursor[1] - exit_pos[1])
    if env.moves_remaining > required_moves and env.grid[cursor[0], cursor[1]] != 0:
        group = env._find_matching_group(cursor[0], cursor[1])
        if len(group) >= 2:
            return [0, 1, 0]
    
    if cursor[1] > exit_pos[1]:
        return [1, 0, 0]
    elif cursor[0] < exit_pos[0]:
        return [4, 0, 0]
    elif cursor[0] > exit_pos[0]:
        return [3, 0, 0]
    return [0, 0, 0]