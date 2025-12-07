def policy(env):
    # Strategy: Maximize forward speed while staying centered in track to avoid boundaries and obstacles. 
    # Use boost only when obstacles are detected ahead to gain bonus rewards and avoid penalties from ineffective boosts.
    obs = env.render()
    height, width = obs.shape[:2]
    kart_x = width // 4  # Kart is fixed at 1/4 screen width due to camera offset
    
    # Find track boundaries at kart's x-position
    top_bound, bot_bound = None, None
    for y in range(height):
        r, g, b = obs[y, kart_x]
        if 180 <= r <= 220 and 180 <= g <= 220 and 200 <= b <= 240:  # Track boundary color
            if top_bound is None:
                top_bound = y
            bot_bound = y
    if top_bound is None:
        top_bound, bot_bound = height//2 - 90, height//2 + 90
    center_y = (top_bound + bot_bound) // 2
    
    # Find kart position (red pixels near left center)
    red_ys = []
    for x in range(kart_x - 10, kart_x + 11):
        for y in range(max(0, center_y-30), min(height, center_y+31)):
            r, g, b = obs[y, x]
            if r > 200 and g < 100 and b < 100:  # Kart red color
                red_ys.append(y)
    kart_y = sum(red_ys) // len(red_ys) if red_ys else center_y
    
    # Steer toward track center with hysteresis to prevent oscillation
    error = kart_y - center_y
    if error > 8:
        a0 = 3  # steer up
    elif error < -8:
        a0 = 4  # steer down
    else:
        a0 = 1  # accelerate forward
    
    # Check for obstacles in danger zone (ahead of kart)
    has_obstacle = False
    for x in range(kart_x + 50, kart_x + 101):
        for y in range(top_bound, bot_bound+1):
            r, g, b = obs[y, x]
            if 200 <= r <= 240 and 30 <= g <= 70 and 30 <= b <= 70:  # Obstacle red
                has_obstacle = True
                break
        if has_obstacle:
            break
    
    a1 = 1 if has_obstacle else 0  # boost only when obstacle detected
    a2 = 0  # secondary action unused
    return [a0, a1, a2]