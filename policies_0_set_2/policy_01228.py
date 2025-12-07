def policy(env):
    # Strategy: Maximize score by shooting aliens while avoiding projectiles. Always fire (a1=1) and never use secondary (a2=0).
    # Move to dodge incoming projectiles first, then align with alien clusters for efficient shooting.
    obs = env._get_observation()
    player_x = 320  # Default center if not found
    proj_xs = []
    alien_xs = []
    
    # Scan for player (green triangle) near bottom
    for y in range(390, 350, -1):
        for x in range(640):
            r, g, b = obs[y, x]
            if abs(r - 0) < 20 and abs(g - 255) < 20 and abs(b - 128) < 20:
                player_x = x
                break
        if player_x != 320:
            break
    
    # Scan for projectiles (yellow) and aliens (pink) above player
    for y in range(200, 350, 5):
        for x in range(0, 640, 5):
            r, g, b = obs[y, x]
            if abs(r - 255) < 20 and abs(g - 200) < 20 and abs(b - 0) < 20:
                proj_xs.append(x)
            elif abs(r - 255) < 20 and abs(g - 64) < 20 and abs(b - 128) < 20:
                alien_xs.append(x)
    
    # Dodge projectiles if any are within danger zone
    danger_projs = [x for x in proj_xs if abs(x - player_x) < 25]
    if danger_projs:
        avg_proj_x = sum(danger_projs) / len(danger_projs)
        move = 4 if avg_proj_x < player_x else 3
    # Otherwise target alien clusters
    elif alien_xs:
        avg_alien_x = sum(alien_xs) / len(alien_xs)
        if avg_alien_x < player_x - 15:
            move = 3
        elif avg_alien_x > player_x + 15:
            move = 4
        else:
            move = 0
    else:
        move = 0
    
    return [move, 1, 0]