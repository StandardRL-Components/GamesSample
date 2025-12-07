def policy(env):
    # Strategy: Move towards the nearest coin while avoiding obstacles. Prioritize coin collection
    # since it gives +10 reward and advances the win condition. Use Euclidean distance to evaluate
    # candidate movements, balancing coin proximity and obstacle avoidance. Secondary actions (a1, a2)
    # are unused in this environment and set to 0.
    import math
    import numpy as np
    
    obs = env._get_observation()
    screen_height, screen_width, _ = obs.shape
    player_size = env.PLAYER_SIZE
    player_speed = env.PLAYER_SPEED
    
    # Define color thresholds (RGB)
    coin_color = np.array([241, 196, 15])
    obstacle_color = np.array([231, 76, 60])
    player_color = np.array([52, 152, 219])
    tolerance = 20
    
    # Find player position (median of player-colored pixels)
    player_pixels = np.where(np.all(np.abs(obs - player_color) < tolerance, axis=2))
    if len(player_pixels[0]) > 0:
        player_y = int(np.median(player_pixels[0]))
        player_x = int(np.median(player_pixels[1]))
    else:
        player_x, player_y = screen_width // 2, screen_height // 2
    
    # Find coins and obstacles
    coin_mask = np.all(np.abs(obs - coin_color) < tolerance, axis=2)
    obstacle_mask = np.all(np.abs(obs - obstacle_color) < tolerance, axis=2)
    coin_y, coin_x = np.where(coin_mask)
    obstacle_y, obstacle_x = np.where(obstacle_mask)
    
    # Calculate distances to nearest coin and obstacle
    min_coin_dist = float('inf')
    nearest_coin = None
    for cx, cy in zip(coin_x, coin_y):
        dist = math.hypot(cx - player_x, cy - player_y)
        if dist < min_coin_dist:
            min_coin_dist = dist
            nearest_coin = (cx, cy)
    
    min_obstacle_dist = float('inf')
    nearest_obstacle = None
    for ox, oy in zip(obstacle_x, obstacle_y):
        dist = math.hypot(ox - player_x, oy - player_y)
        if dist < min_obstacle_dist:
            min_obstacle_dist = dist
            nearest_obstacle = (ox, oy)
    
    # Evaluate candidate movements (0: none, 1: up, 2: down, 3: left, 4: right)
    best_score = -float('inf')
    best_action = 0
    for move in range(5):
        dx, dy = 0, 0
        if move == 1: dy = -player_speed
        elif move == 2: dy = player_speed
        elif move == 3: dx = -player_speed
        elif move == 4: dx = player_speed
        
        new_x = max(player_size, min(screen_width - player_size, player_x + dx))
        new_y = max(player_size, min(screen_height - player_size, player_y + dy))
        
        # Score based on coin proximity and obstacle avoidance
        coin_score = 0
        if nearest_coin:
            coin_dist = math.hypot(nearest_coin[0] - new_x, nearest_coin[1] - new_y)
            coin_score = -coin_dist  # Closer is better
        
        obstacle_score = 0
        if nearest_obstacle and min_obstacle_dist < 100:  # Only avoid nearby obstacles
            obstacle_dist = math.hypot(nearest_obstacle[0] - new_x, nearest_obstacle[1] - new_y)
            obstacle_score = obstacle_dist  # Farther is better
        
        score = coin_score + 0.5 * obstacle_score
        if score > best_score:
            best_score = score
            best_action = move
    
    return [best_action, 0, 0]