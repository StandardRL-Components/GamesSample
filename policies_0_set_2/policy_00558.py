def policy(env):
    # Strategy: Prioritize attacking nearest monster when in range to maximize reward from kills and combos.
    # Move towards nearest monster or coin if no monsters, using screen coordinates and isometric movement vectors.
    # Always attack (a1=1) to exploit cooldown and earn rewards, and ignore secondary action (a2=0).
    import pygame.surfarray
    import numpy as np

    # Get current screen observation
    arr = pygame.surfarray.array3d(env.screen)
    obs = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    # Define color thresholds for player (green) and monsters (red)
    green_low = np.array([20, 225, 120])
    green_high = np.array([80, 285, 180])
    red_low = np.array([225, 50, 50])
    red_high = np.array([285, 110, 110])
    yellow_low = np.array([225, 200, 20])
    yellow_high = np.array([285, 240, 80])
    
    # Find player position (green circle near center)
    center_region = obs[120:280, 220:420]
    green_mask = np.all(center_region >= green_low, axis=-1) & np.all(center_region <= green_high, axis=-1)
    player_y, player_x = np.where(green_mask)
    if len(player_y) > 0:
        player_pos = (np.mean(player_x) + 220, np.mean(player_y) + 120)
    else:
        return [0, 1, 0]  # Default to no movement if player not found
    
    # Find monsters (red rectangles) and coins (yellow circles)
    red_mask = np.all(obs >= red_low, axis=-1) & np.all(obs <= red_high, axis=-1)
    yellow_mask = np.all(obs >= yellow_low, axis=-1) & np.all(obs <= yellow_high, axis=-1)
    monster_y, monster_x = np.where(red_mask)
    coin_y, coin_x = np.where(yellow_mask)
    
    # Calculate distances to all monsters/coins
    if len(monster_x) > 0:
        dists = (monster_x - player_pos[0])**2 + (monster_y - player_pos[1])**2
        target_idx = np.argmin(dists)
        target_pos = (monster_x[target_idx], monster_y[target_idx])
    elif len(coin_x) > 0:
        dists = (coin_x - player_pos[0])**2 + (coin_y - player_pos[1])**2
        target_idx = np.argmin(dists)
        target_pos = (coin_x[target_idx], coin_y[target_idx])
    else:
        return [0, 1, 0]  # No targets found
    
    # Calculate direction vector to target
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    
    # Project onto isometric movement vectors (right: [24,12], left: [-24,-12], up: [24,-12], down: [-24,12])
    dots = [
        dx*24 + dy*12,   # right
        dx*(-24) + dy*(-12), # left
        dx*24 + dy*(-12), # up
        dx*(-24) + dy*12   # down
    ]
    move_id = np.argmax(dots) + 1  # Convert to action ID (1-4)
    
    return [move_id, 1, 0]