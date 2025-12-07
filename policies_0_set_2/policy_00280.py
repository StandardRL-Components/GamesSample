def policy(env):
    # Avoid asteroids by moving away from the nearest and most dangerous ones (considering velocity and proximity).
    # This maximizes survival time by minimizing collision risk while collecting survival rewards.
    total_force_x, total_force_y = 0.0, 0.0
    for asteroid in env.asteroids:
        dx = asteroid['pos'][0] - env.player_pos[0]
        dy = asteroid['pos'][1] - env.player_pos[1]
        dx_wrapped = dx - env.SCREEN_WIDTH * round(dx / env.SCREEN_WIDTH)
        dy_wrapped = dy - env.SCREEN_HEIGHT * round(dy / env.SCREEN_HEIGHT)
        dist_sq = dx_wrapped*dx_wrapped + dy_wrapped*dy_wrapped
        if dist_sq < 1e-5:
            continue
        total_force_x -= dx_wrapped / dist_sq
        total_force_y -= dy_wrapped / dist_sq
        
    force_sq = total_force_x*total_force_x + total_force_y*total_force_y
    if force_sq < 0.01:
        a0 = 0
    elif abs(total_force_x) > abs(total_force_y):
        a0 = 4 if total_force_x > 0 else 3
    else:
        a0 = 2 if total_force_y > 0 else 1
        
    return [a0, 1, 0]