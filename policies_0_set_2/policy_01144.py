def policy(env):
    # Strategy: Prioritize survival by dodging immediate threats (projectiles and nearby aliens),
    # then align with alien clusters to maximize firing efficiency while maintaining safe distance.
    # Always fire when not dodging to maintain constant damage output.
    player_x = env.player_pos.x
    player_y = env.player_pos.y
    danger_zone = 80  # Vertical distance to consider threats
    safe_distance = 40  # Horizontal buffer from threats
    
    # Collect immediate threats (projectiles and aliens in danger zone)
    threats = []
    for proj in env.alien_projectiles:
        if player_y - proj.y < danger_zone and abs(proj.x - player_x) < safe_distance:
            threats.append(('projectile', proj.x))
    for alien in env.aliens:
        if player_y - alien['pos'].y < danger_zone and abs(alien['pos'].x - player_x) < safe_distance:
            threats.append(('alien', alien['pos'].x))
    
    # Dodge threats by moving away from closest danger
    if threats:
        closest_x = min(threats, key=lambda x: abs(x[1] - player_x))[1]
        move = 4 if closest_x < player_x else 3
    # If no threats, align with largest alien cluster for efficient shooting
    elif env.aliens:
        # Group aliens by horizontal proximity and target largest cluster
        clusters = {}
        for alien in env.aliens:
            cluster_key = round(alien['pos'].x / 50)
            clusters[cluster_key] = clusters.get(cluster_key, 0) + 1
        if clusters:
            target_cluster = max(clusters, key=clusters.get) * 50
            if target_cluster < player_x - 10:
                move = 3
            elif target_cluster > player_x + 10:
                move = 4
            else:
                move = 0
        else:
            move = 0
    else:
        move = 0
    
    return [move, 1, 0]  # Always fire when possible, no secondary action