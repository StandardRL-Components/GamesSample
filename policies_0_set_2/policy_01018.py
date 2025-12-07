def policy(env):
    """
    Strategy: Prioritize collecting items while avoiding ghosts. If a ghost is too close, move away.
    Otherwise, move towards the nearest item or exit if all items are collected. Use dot product
    to select the movement direction that best aligns with the desired vector, avoiding walls.
    """
    import math
    import pygame

    danger_radius = 50
    player_pos = env.player_pos
    player_radius = env.PLAYER_RADIUS
    player_speed = env.PLAYER_SPEED

    # Find closest ghost
    min_ghost_dist = float('inf')
    closest_ghost = None
    for ghost in env.ghosts:
        gx, gy = ghost['pos']
        dist = math.hypot(player_pos[0] - gx, player_pos[1] - gy)
        if dist < min_ghost_dist:
            min_ghost_dist = dist
            closest_ghost = ghost

    avoiding_ghost = (min_ghost_dist < danger_radius)

    if avoiding_ghost:
        # Move away from closest ghost
        gx, gy = closest_ghost['pos']
        dx = player_pos[0] - gx
        dy = player_pos[1] - gy
        norm = math.hypot(dx, dy)
        if norm == 0:
            norm = 1
        dx /= norm
        dy /= norm
    else:
        # Move towards target (nearest item or exit)
        if len(env.items) > 0:
            min_dist = float('inf')
            target = None
            for item in env.items:
                ix, iy = item.centerx, item.centery
                dist = math.hypot(player_pos[0] - ix, player_pos[1] - iy)
                if dist < min_dist:
                    min_dist = dist
                    target = (ix, iy)
        else:
            target = (env.exit_rect.centerx, env.exit_rect.centery)
        dx = target[0] - player_pos[0]
        dy = target[1] - player_pos[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            norm = 1
        dx /= norm
        dy /= norm

    # Evaluate each movement action
    best_action = 0
    best_score = -10e9
    movements = [(0, 0, 0), (0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]

    for move_dx, move_dy, action_idx in movements:
        candidate_x = player_pos[0] + move_dx * player_speed
        candidate_y = player_pos[1] + move_dy * player_speed
        candidate_rect = pygame.Rect(0, 0, player_radius * 2, player_radius * 2)
        candidate_rect.center = (candidate_x, candidate_y)
        
        if candidate_rect.collidelist(env.walls) != -1:
            score = -10e9
        else:
            score = move_dx * dx + move_dy * dy

        if score > best_score:
            best_score = score
            best_action = action_idx

    return [best_action, 0, 0]