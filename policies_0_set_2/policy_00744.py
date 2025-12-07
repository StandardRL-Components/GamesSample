def policy(env):
    # Navigate towards nearest uncollected asteroid while avoiding mines using potential fields.
    # Prioritize asteroid collection (high reward) while maintaining safe distance from mines (high penalty).
    # Always use boost for efficiency and ignore secondary action (unused in this environment).
    player_pos = env.player_pos
    closest_ast = None
    min_ast_dist = float('inf')
    for ast in env.asteroids:
        if not ast['collected']:
            dx = ast['pos'][0] - player_pos[0]
            dy = ast['pos'][1] - player_pos[1]
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < min_ast_dist:
                min_ast_dist = dist
                closest_ast = ast

    closest_mine = None
    min_mine_dist = float('inf')
    for mine in env.mines:
        dx = mine['pos'][0] - player_pos[0]
        dy = mine['pos'][1] - player_pos[1]
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < min_mine_dist:
            min_mine_dist = dist
            closest_mine = mine

    att_dx, att_dy = 0, 0
    if closest_ast is not None:
        dx = closest_ast['pos'][0] - player_pos[0]
        dy = closest_ast['pos'][1] - player_pos[1]
        norm = (dx*dx + dy*dy) ** 0.5
        if norm > 0:
            att_dx = dx / norm
            att_dy = dy / norm

    rep_dx, rep_dy = 0, 0
    if closest_mine is not None:
        dx = player_pos[0] - closest_mine['pos'][0]
        dy = player_pos[1] - closest_mine['pos'][1]
        norm = (dx*dx + dy*dy) ** 0.5
        if norm > 0:
            rep_dx = dx / norm
            rep_dy = dy / norm
        rep_weight = 3.0 if min_mine_dist < 50 else 1.5
        rep_dx *= rep_weight
        rep_dy *= rep_weight

    total_dx = att_dx + rep_dx
    total_dy = att_dy + rep_dy

    if abs(total_dx) > abs(total_dy):
        movement = 4 if total_dx > 0 else 3
    else:
        movement = 2 if total_dy > 0 else 1

    if (total_dx**2 + total_dy**2) < 0.01:
        movement = 0

    return [movement, 1, 0]