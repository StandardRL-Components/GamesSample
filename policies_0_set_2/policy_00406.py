def policy(env):
    # This policy maximizes reward by prioritizing avoiding asteroids and collecting power-ups.
    # It moves towards power-ups when safe and uses shields only when collisions are imminent.
    # It avoids the bottom half to reduce penalty and uses score power-ups immediately.
    player_x, player_y = env.player_pos
    current_vel = env.player_vel
    screen_height = env.screen_height
    player_radius = env.PLAYER_RADIUS
    accel = env.PLAYER_ACCEL
    max_speed = env.PLAYER_MAX_SPEED
    speed_multiplier = 2.0 if "speed" in env.active_powerups else 1.0

    closest_asteroid = None
    min_asteroid_dist = float('inf')
    for asteroid in env.asteroids:
        if asteroid['pos'][0] > player_x:
            dist = asteroid['pos'][0] - player_x
            if dist < min_asteroid_dist:
                min_asteroid_dist = dist
                closest_asteroid = asteroid

    closest_powerup = None
    min_powerup_dist = float('inf')
    for powerup in env.powerups:
        if powerup['pos'][0] > player_x:
            dist = powerup['pos'][0] - player_x
            if dist < min_powerup_dist:
                min_powerup_dist = dist
                closest_powerup = powerup

    best_score = -10**9
    best_action = 0
    for action in [0, 1, 2]:
        if action == 0:
            new_vel = current_vel
        elif action == 1:
            new_vel = current_vel - accel
        else:
            new_vel = current_vel + accel

        new_vel = max(-max_speed, min(max_speed, new_vel))
        new_y = player_y + new_vel * speed_multiplier
        new_y = max(player_radius, min(screen_height - player_radius, new_y))

        score = 0
        if closest_asteroid is not None:
            asteroid_y = closest_asteroid['pos'][1]
            asteroid_radius = closest_asteroid['radius']
            vertical_dist = abs(new_y - asteroid_y)
            desired_gap = asteroid_radius + player_radius
            if vertical_dist < desired_gap:
                score -= (desired_gap - vertical_dist) * 10

        if closest_powerup is not None:
            powerup_y = closest_powerup['pos'][1]
            score -= abs(new_y - powerup_y)

        if new_y > screen_height / 2:
            score -= 10

        if score > best_score:
            best_score = score
            best_action = action

    held = env.held_powerup
    activate = False
    if held == "score":
        activate = True
    elif held == "shield":
        for asteroid in env.asteroids:
            dx = asteroid['pos'][0] - player_x
            dy = asteroid['pos'][1] - player_y
            threshold = asteroid['radius'] + player_radius + 5
            if dx > 0 and dx < threshold and abs(dy) < threshold:
                if dx*dx + dy*dy < threshold*threshold:
                    activate = True
                    break

    return [best_action, 1 if activate else 0, 0]