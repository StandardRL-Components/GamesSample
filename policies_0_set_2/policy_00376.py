def policy(env):
    """
    Maximizes score by jumping only when necessary to clear obstacles, avoiding safe jump penalties (-0.2)
    and earning risky jump rewards (+1.0). Uses adaptive threshold (5 * speed) to time jumps accurately
    as game speed increases. Avoids no-ops by checking is_jumping state and obstacle proximity.
    """
    if env.is_jumping:
        return [0, 0, 0]
    threshold = 5 * env.runner_speed
    for obstacle in env.obstacles:
        if obstacle['cleared']:
            continue
        dist = obstacle['rect'].right - env.player_rect.left
        if 0 <= dist < threshold:
            return [1, 0, 0]
    return [0, 0, 0]