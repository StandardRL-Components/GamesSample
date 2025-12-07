def policy(env):
    # Strategy: Move away from the nearest zombie using inverse square repulsion to maximize survival time.
    # This prioritizes immediate evasion (reward per step) and avoids collisions (large negative reward).
    repulsion_x, repulsion_y = 0.0, 0.0
    for zombie in env.zombies:
        dx = env.player_pos[0] - zombie.centerx
        dy = env.player_pos[1] - zombie.centery
        dist_sq = dx*dx + dy*dy
        if dist_sq < 1e-5:
            repulsion_x += dx
            repulsion_y += dy
        else:
            repulsion_x += dx / dist_sq
            repulsion_y += dy / dist_sq
            
    if repulsion_x == 0 and repulsion_y == 0:
        movement = 0
    else:
        if abs(repulsion_x) > abs(repulsion_y):
            movement = 4 if repulsion_x > 0 else 3
        else:
            movement = 2 if repulsion_y > 0 else 1
            
    return [movement, 0, 0]