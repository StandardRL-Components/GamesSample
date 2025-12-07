def policy(env):
    # Prioritize attacking adjacent enemies (+5 reward), then move toward potions if health < max, else move toward exit.
    # Avoid moving into enemies and break ties by minimizing Manhattan distance to target.
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    enemies = env.enemies
    potions = env.potions
    health = env.player_health
    max_health = env.MAX_HEALTH

    # Check for adjacent enemies to attack
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        if (player_pos[0] + dx, player_pos[1] + dy) in enemies:
            return [0, 1, 0]  # Attack without moving

    # Determine target: nearest potion if health < max, else exit
    if health < max_health and potions:
        target = min(potions, key=lambda p: abs(p[0]-player_pos[0]) + abs(p[1]-player_pos[1]))
    else:
        target = exit_pos

    # Find move that minimizes distance to target without stepping on enemies
    best_move = 0
    best_dist = abs(target[0]-player_pos[0]) + abs(target[1]-player_pos[1])
    for move, (dx, dy) in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)]):
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        if (0 <= new_pos[0] < env.GRID_WIDTH and 0 <= new_pos[1] < env.GRID_HEIGHT and 
            new_pos not in enemies):
            new_dist = abs(target[0]-new_pos[0]) + abs(target[1]-new_pos[1])
            if new_dist < best_dist:
                best_dist = new_dist
                best_move = move
    return [best_move, 0, 0]