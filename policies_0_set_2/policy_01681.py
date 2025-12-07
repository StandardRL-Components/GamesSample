def policy(env):
    # Greedily move towards the nearest target to maximize collection rate and minimize time penalty.
    # Uses Manhattan distance and prioritizes moving actions over no-op to avoid wasting steps.
    if len(env.targets) == 0:
        return [0, 0, 0]
    if env.player_pos in env.targets:
        return [0, 0, 0]
    player_x, player_y = env.player_pos
    nearest_target = min(env.targets, key=lambda t: abs(t[0]-player_x) + abs(t[1]-player_y))
    actions = [1, 2, 3, 4, 0]
    best_action = actions[0]
    best_dist = float('inf')
    for a in actions:
        x, y = player_x, player_y
        if a == 1: y -= 1
        elif a == 2: y += 1
        elif a == 3: x -= 1
        elif a == 4: x += 1
        x = max(0, min(env.GRID_WIDTH-1, x))
        y = max(0, min(env.GRID_HEIGHT-1, y))
        dist = abs(nearest_target[0]-x) + abs(nearest_target[1]-y)
        if dist < best_dist:
            best_dist = dist
            best_action = a
    return [best_action, 0, 0]