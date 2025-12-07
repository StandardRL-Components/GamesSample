def policy(env):
    # Strategy: Focus on upward progression by jumping toward platforms directly above the player.
    # Prioritize vertical jumps when platforms are centered, otherwise use directional jumps
    # to align with nearest platforms while minimizing horizontal movement to save time.
    obs = env._get_observation()
    player_y = 280
    cyan = [50, 255, 255]
    tol = 30
    player_x = 320
    for x in range(640):
        if all(abs(obs[player_y, x, i] - cyan[i]) < tol for i in range(3)):
            player_x = x
            break

    platform_xs = []
    for y in range(250, 50, -5):
        for x in range(max(0, player_x-100), min(640, player_x+100), 5):
            if sum(obs[y, x]) > 100 and all(abs(obs[y, x, i] - cyan[i]) > tol for i in range(3)):
                platform_xs.append(x)
        if platform_xs:
            break

    if not platform_xs:
        return [1, 0, 0]

    avg_x = sum(platform_xs) // len(platform_xs)
    if abs(avg_x - player_x) < 15:
        return [1, 0, 0]
    elif avg_x < player_x:
        return [3, 0, 0]
    else:
        return [4, 0, 0]