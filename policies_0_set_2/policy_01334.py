def policy(env):
    # Strategy: Maximize reward by continuously firing and aligning with alien formations.
    # Move towards the center of visible aliens to increase hit probability while avoiding edges.
    # Always fire (a1=1) to maximize alien destruction and reward, ignoring cooldown (env handles cooldown).
    obs = env._get_observation()
    alien_x_sum = 0
    alien_count = 0
    for y in range(50, 300, 20):
        for x in range(0, 640, 10):
            r, g, b = obs[y, x, 0], obs[y, x, 1], obs[y, x, 2]
            if r > 200 and g < 100 and b < 100:  # Detect red alien pixels
                alien_x_sum += x
                alien_count += 1
    if alien_count > 0:
        avg_x = alien_x_sum / alien_count
        player_x = 320  # Assume center if not detected
        for y in range(360, 400):
            for x in range(0, 640):
                r, g, b = obs[y, x, 0], obs[y, x, 1], obs[y, x, 2]
                if g > 200 and r < 100 and b < 100:  # Detect green player pixels
                    player_x = x
                    break
        if avg_x < player_x - 15:
            movement = 3  # Left
        elif avg_x > player_x + 15:
            movement = 4  # Right
        else:
            movement = 0  # None
    else:
        movement = 0  # No aliens visible
    return [movement, 1, 0]  # Always fire, no secondary action