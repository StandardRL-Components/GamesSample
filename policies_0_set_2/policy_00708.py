def policy(env):
    # Strategy: Project sled trajectory to land 100px ahead of current track end, then choose track angle
    # that minimizes vertical deviation. Prefer downward slopes to maintain speed and avoid stalling.
    slopes = [0.0875, -1.0, 1.0, -0.2679, 0.2679]  # tan(5°), tan(-45°), tan(45°), tan(-15°), tan(15°)
    last_point = env.track_points[-1]
    look_ahead_x = last_point.x + 100
    if env.sled_vel.x <= 0:
        best_action = 2  # Steep down to regain momentum if stalled
    else:
        t = (look_ahead_x - env.sled_pos.x) / env.sled_vel.x
        proj_y = env.sled_pos.y + env.sled_vel.y * t + 0.5 * env.GRAVITY * t * t
        proj_y = max(0, min(env.HEIGHT, proj_y))
        dx = look_ahead_x - last_point.x
        dy = proj_y - last_point.y
        if abs(dx) < 1e-6:
            best_action = 2 if dy > 0 else 1
        else:
            desired_slope = dy / dx
            differences = [abs(desired_slope - s) for s in slopes]
            best_action = differences.index(min(differences))
    return [best_action, 0, 0]