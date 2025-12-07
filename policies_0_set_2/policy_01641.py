def policy(env):
    # Strategy: Move right consistently to reach flag while collecting coins. Jump when needed to avoid pits or reach elevated coins.
    # Prioritize immediate movement toward right-side targets (coins/flag) while avoiding pits through timed jumps.
    player_x, player_y = env.player_pos
    is_grounded = env.is_grounded
    a0, a1 = 0, 0

    # Always move right unless blocked or need precise positioning
    a0 = 4  # right

    # Check if pit ahead and jump when approaching edge
    for pit in env.pits:
        if player_x + 24 >= pit.left - 5 and player_x <= pit.left + 10 and is_grounded:
            a1 = 1
            break

    # Jump for coins significantly above player
    for coin in env.coins:
        if coin.centery < player_y - 30 and abs(coin.centerx - player_x) < 50 and is_grounded:
            a1 = 1
            break

    return [a0, a1, 0]