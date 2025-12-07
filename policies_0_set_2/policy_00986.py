def policy(env):
    # Strategy: Always move right to reach the checkpoint quickly. Jump when gaps are detected below or ahead to avoid falling.
    # Check player's vertical position and nearby pixels to determine if a jump is needed to continue progressing right.
    obs = env._get_observation()
    player_y = None
    for y in range(399, 100, -1):
        r, g, b = obs[y, 213]
        if 40 <= r <= 80 and 140 <= g <= 180 and 235 <= b <= 275:  # Player blue color
            player_y = y
            break
    if player_y is None:
        return [4, 0, 0]
    on_ground = any(
        abs(obs[player_y + 5, 213 + dx][0] - 120) < 20 and
        abs(obs[player_y + 5, 213 + dx][1] - 130) < 20 and
        abs(obs[player_y + 5, 213 + dx][2] - 150) < 20
        for dx in [-10, 0, 10]
    )
    gap_ahead = not any(
        abs(obs[player_y + 5, 233 + dx][0] - 120) < 20 and
        abs(obs[player_y + 5, 233 + dx][1] - 130) < 20 and
        abs(obs[player_y + 5, 233 + dx][2] - 150) < 20
        for dx in [-5, 0, 5]
    )
    if on_ground and gap_ahead:
        return [4, 1, 0]
    return [4, 0, 0]