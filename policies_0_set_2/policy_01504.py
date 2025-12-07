def policy(env):
    # Strategy: Maximize speed and avoid obstacles by analyzing the road ahead in the observation.
    # Accelerate and boost when the path is clear, and turn to avoid obstacles. Avoid safe penalties by maintaining speed.
    obs = env._get_observation()
    obstacle_color1 = (255, 0, 100)
    obstacle_color2 = (200, 0, 80)
    left_count, right_count = 0, 0
    for i in range(0, 100, 5):
        for j in range(70, 570, 5):
            r, g, b = obs[i, j]
            if (abs(r - obstacle_color1[0]) < 10 and abs(g - obstacle_color1[1]) < 10 and abs(b - obstacle_color1[2]) < 10) or \
               (abs(r - obstacle_color2[0]) < 10 and abs(g - obstacle_color2[1]) < 10 and abs(b - obstacle_color2[2]) < 10):
                if j < 320:
                    left_count += 1
                else:
                    right_count += 1
    current_speed = env.player_speed
    if left_count > 5 or right_count > 5:
        if left_count > right_count:
            movement = 4
        else:
            movement = 3
        boost = 0
    else:
        movement = 1
        boost = 1 if current_speed < env.BOOST_SPEED else 0
    brake = 0
    return [movement, boost, brake]