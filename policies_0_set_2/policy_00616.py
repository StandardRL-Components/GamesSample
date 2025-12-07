def policy(env):
    # Strategy: Always move right to maximize progress reward and reach the flag quickly.
    # Jump when grounded to collect coins above or avoid falling off platforms.
    # Use internal state (read-only) to detect coins/platform edges for optimal jumping.
    action = [4, 0, 0]  # Default: move right
    if env.is_grounded:
        # Check for collectible coins above and slightly ahead
        player_x, player_y = env.player_pos
        for coin in env.coins:
            dx = coin.centerx - player_x
            dy = player_y - coin.centery
            if 0 < dx < 80 and 0 < dy < 100:
                action[0] = 1  # Jump to collect coin
                break
        # Check if near platform edge
        player_rect = (player_x, player_y, env.PLAYER_SIZE, env.PLAYER_SIZE)
        for platform in env.platforms:
            if (player_rect[0] >= platform.left and player_rect[0] <= platform.right and
                player_rect[1] + player_rect[3] >= platform.top and
                player_rect[1] + player_rect[3] <= platform.bottom):
                if platform.right - player_x < 25:  # Near right edge
                    action[0] = 1  # Jump to avoid falling
                break
    return action