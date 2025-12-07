def policy(env):
    # Strategy: Survive by kiting zombies while shooting continuously. Always shoot (a1=1) to maximize kills and reduce threats. 
    # Move away from the nearest zombie to avoid damage. Dash (a2=1) is omitted due to state limitations and cooldown complexity.
    obs = env._get_observation()
    h, w, _ = obs.shape
    step = 4
    player_pixels = []
    zombie_pixels = []
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = obs[y, x]
            if r == 0 and g == 255 and b == 127:
                player_pixels.append((y, x))
            if (r == 255 and g == 69 and b == 0) or (r == 255 and g == 255 and b == 255):
                zombie_pixels.append((y, x))
                
    if player_pixels:
        py = sum(p[0] for p in player_pixels) / len(player_pixels)
        px = sum(p[1] for p in player_pixels) / len(player_pixels)
    else:
        py, px = h / 2, w / 2
        
    if zombie_pixels:
        min_dist = float('inf')
        closest = None
        for z in zombie_pixels:
            dist = (z[0] - py)**2 + (z[1] - px)**2
            if dist < min_dist:
                min_dist = dist
                closest = z
        dy, dx = py - closest[0], px - closest[1]
        mag = (dy**2 + dx**2)**0.5
        if mag < 1e-5:
            move = 0
        else:
            dy, dx = dy / mag, dx / mag
            if abs(dy) > abs(dx):
                move = 2 if dy > 0 else 1
            else:
                move = 4 if dx > 0 else 3
    else:
        move = 1
        
    return [move, 1, 0]