def policy(env):
    # Strategy: Move towards the nearest unsquashed bug to maximize points and time efficiency.
    # If on a bug, squash it. Otherwise, choose the movement that minimizes Euclidean distance to the nearest bug.
    current_pos = env.cursor_pos
    dist = env._get_dist_to_nearest_bug(current_pos)
    if dist is None:
        return [0, 0, 0]  # No bugs left
    if dist == 0:
        return [0, 1, 0]  # Squash bug under cursor
    best_move = 0
    best_dist = dist
    moves = [0, 1, 2, 3, 4]
    for move in moves:
        candidate_pos = list(current_pos)
        if move == 1:
            candidate_pos[1] -= 1
        elif move == 2:
            candidate_pos[1] += 1
        elif move == 3:
            candidate_pos[0] -= 1
        elif move == 4:
            candidate_pos[0] += 1
        candidate_pos[0] = max(0, min(env.GRID_WIDTH - 1, candidate_pos[0]))
        candidate_pos[1] = max(0, min(env.GRID_HEIGHT - 1, candidate_pos[1]))
        candidate_dist = env._get_dist_to_nearest_bug(candidate_pos)
        if candidate_dist is not None and candidate_dist < best_dist:
            best_dist = candidate_dist
            best_move = move
    return [best_move, 0, 0]