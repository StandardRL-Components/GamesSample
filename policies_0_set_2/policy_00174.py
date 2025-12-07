def policy(env):
    # Strategy: Avoid zombies by moving away from the nearest one, while moving towards the nearest coin when safe.
    # This balances survival (avoiding negative reward) with collecting coins (positive reward).
    px, py = env.player_rect.center
    nearest_zombie = None
    min_zombie_dist = float('inf')
    for zombie in env.zombies:
        zx, zy = zombie.center
        dist = ((px - zx) ** 2 + (py - zy) ** 2) ** 0.5
        if dist < min_zombie_dist:
            min_zombie_dist = dist
            nearest_zombie = (zx, zy)
    
    nearest_coin = None
    min_coin_dist = float('inf')
    for coin in env.coins:
        cx, cy = coin.center
        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if dist < min_coin_dist:
            min_coin_dist = dist
            nearest_coin = (cx, cy)
    
    if nearest_zombie and min_zombie_dist < 100:
        zx, zy = nearest_zombie
        dx, dy = px - zx, py - zy
        if abs(dx) > abs(dy):
            return [3 if dx < 0 else 4, 0, 0]
        else:
            return [1 if dy < 0 else 2, 0, 0]
    elif nearest_coin:
        cx, cy = nearest_coin
        dx, dy = cx - px, cy - py
        if abs(dx) > abs(dy):
            return [3 if dx < 0 else 4, 0, 0]
        else:
            return [1 if dy < 0 else 2, 0, 0]
    else:
        return [0, 0, 0]