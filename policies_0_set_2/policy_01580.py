def policy(env):
    # Strategy: Prioritize moving right to reach the end flag for stage completion rewards.
    # Jump to collect coins (immediate reward) and avoid pits by jumping when detected ahead.
    # Use ground state to prevent mid-air jumps and maintain momentum for efficiency.
    action = [4, 0, 0]  # Default: move right
    player_x, player_y = env.player_pos.x, env.player_pos.y
    if env.on_ground:
        # Check for nearby coins above player to jump and collect
        for coin in env.coins:
            dx = coin.centerx - player_x
            dy = coin.centery - player_y
            if 0 < dx < 50 and -50 < dy < 0:
                action = [1, 0, 0]  # Jump for coin
                break
        # Check for pits immediately ahead to jump over
        for pit in env.pits:
            if pit.left - player_x < 50 and pit.left > player_x and abs(pit.top - (player_y + 16)) < 5:
                action = [1, 0, 0]  # Jump over pit
                break
    return action