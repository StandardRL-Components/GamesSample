def policy(env):
    # Strategy: Avoid asteroids by moving away from the nearest detected threat. Prioritize immediate survival by maintaining distance from obstacles.
    obs = env._get_observation()
    player_pixels = []
    asteroid_pixels = []
    for y in range(0, env.SCREEN_HEIGHT, 4):
        for x in range(0, env.SCREEN_WIDTH, 4):
            r, g, b = obs[y, x]
            if r > 250 and g > 250 and b > 250:
                player_pixels.append((x, y))
            elif 110 <= r <= 170 and 110 <= g <= 170 and 120 <= b <= 180:
                asteroid_pixels.append((x, y))
    
    if not player_pixels:
        return [0, 0, 0]
    
    player_x = sum(p[0] for p in player_pixels) / len(player_pixels)
    player_y = sum(p[1] for p in player_pixels) / len(player_pixels)
    
    if not asteroid_pixels:
        return [0, 0, 0]
    
    min_dist_sq = float('inf')
    nearest_ast = None
    for (x, y) in asteroid_pixels:
        dist_sq = (x - player_x)**2 + (y - player_y)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_ast = (x, y)
    
    dx = nearest_ast[0] - player_x
    dy = nearest_ast[1] - player_y
    
    if abs(dx) > abs(dy):
        a0 = 3 if dx > 0 else 4
    else:
        a0 = 2 if dy > 0 else 1
    
    return [a0, 0, 0]