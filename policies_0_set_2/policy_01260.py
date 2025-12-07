def policy(env):
    # Strategy: Jump when on ground to ascend, then air-control toward center to maximize landing chances on next platform.
    # Prioritize vertical progress (reward) by always jumping when grounded, then adjust horizontally to avoid edges.
    obs = env._get_observation()
    player_yellow = (255, 255, 0)
    player_pixels = []
    for y in range(180, 220):
        for x in range(300, 340):
            if tuple(obs[y, x]) == player_yellow:
                player_pixels.append((x, y))
    if not player_pixels:
        return [0, 0, 0]
    player_x = sum(p[0] for p in player_pixels) // len(player_pixels)
    on_ground = any(
        tuple(obs[min(player_y + 15, 399), x]) != (0, 0, 10)
        for x in range(player_x - 10, player_x + 10)
        for player_y in {p[1] for p in player_pixels}
    )
    if on_ground:
        return [1, 0, 0]
    else:
        if player_x < 320:
            return [4, 0, 0]
        else:
            return [3, 0, 0]