def policy(env):
    """
    Strategy: Prioritize collecting items for immediate reward, then head to exit.
    Avoid shadows and choose moves that minimize Manhattan distance to the nearest goal.
    Break ties by direction order (up, down, left, right) to prevent oscillation.
    """
    if env.jump_state['is_jumping']:
        return [0, 0, 0]
    
    px, py = env.player_pos
    items = [item['pos'] for item in env.items]
    shadows = [shadow['pos'] for shadow in env.shadows]
    exit_pos = env.exit_pos
    goals = items if items else [exit_pos]
    
    best_action = 0
    best_score = -min(abs(px - g[0]) + abs(py - g[1]) for g in goals)
    
    for move, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)], start=1):
        nx, ny = px + dx, py + dy
        if not (0 <= nx < 11 and 0 <= ny < 11):
            continue
        if any(nx == s[0] and ny == s[1] for s in shadows):
            score = -100
        else:
            min_dist = min(abs(nx - g[0]) + abs(ny - g[1]) for g in goals)
            score = -min_dist
        if score > best_score:
            best_score = score
            best_action = move
            
    return [best_action, 0, 0]