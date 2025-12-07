def policy(env):
    # Strategy: Jump only when an asteroid is approaching the player's vertical position to avoid collisions.
    # We scan a vertical column ahead of the player (x=100 to 150) for asteroid colors (gray shades).
    # If an asteroid is detected near the player's y-position, jump when on ground to maximize survival reward.
    obs = env._get_observation()
    player_x = 100
    ground_y = 360  # SCREEN_HEIGHT - 40
    player_height = 40
    
    # Estimate player y-position by finding green pixels (player color) near x=100
    player_y = None
    for y in range(ground_y - player_height, ground_y):
        r, g, b = obs[y, player_x]
        if g > 200 and r < 100 and b < 100:  # Green player pixel
            player_y = y
            break
    if player_y is None:
        player_y = ground_y - player_height  # Fallback if not found
    
    # Check for asteroids in the path (x from 100 to 150)
    asteroid_detected = False
    for x in range(player_x, player_x + 50, 5):  # Sample every 5 pixels
        for y in range(max(0, player_y - 20), min(ground_y, player_y + player_height + 20), 5):
            r, g, b = obs[y, x]
            # Check for asteroid colors (gray shades: 100-150)
            if 100 <= r <= 150 and 100 <= g <= 150 and 100 <= b <= 150:
                asteroid_detected = True
                break
        if asteroid_detected:
            break
    
    # Jump only if asteroid detected and player is near ground (on_ground proxy)
    if asteroid_detected and player_y >= ground_y - player_height - 5:
        return [0, 1, 0]  # Jump action
    else:
        return [0, 0, 0]  # No action