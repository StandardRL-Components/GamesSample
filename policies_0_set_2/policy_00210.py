def policy(env):
    # Strategy: Prioritize attacking adjacent enemies when facing them, then move toward exit or gold.
    # This balances combat and progression by eliminating threats first, then optimizing pathfinding.
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    walls = env.walls
    enemies = [e.pos for e in env.enemies if e.health > 0]
    gold_items = env.gold_items

    # Check for adjacent enemies and attack if facing one
    adjacent_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in adjacent_dirs:
        adj_pos = (player_pos[0] + dx, player_pos[1] + dy)
        if adj_pos in enemies:
            if env.player_facing_dir == (dx, dy):
                return [0, 1, 0]  # Attack without moving
            else:
                move_map = {(0, 1): 2, (0, -1): 1, (1, 0): 4, (-1, 0): 3}
                return [move_map[(dx, dy)], 1, 0]  # Turn and attack

    # Move toward exit or gold, avoiding walls and enemies
    best_action, best_score = 0, -float('inf')
    for move in range(5):  # 0=noop, 1=up, 2=down, 3=left, 4=right
        if move == 0:
            new_pos = player_pos
        else:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
            new_pos = (player_pos[0] + dx, player_pos[1] + dy)
            if new_pos in walls or new_pos in enemies:
                continue

        dist_to_exit = abs(new_pos[0] - exit_pos[0]) + abs(new_pos[1] - exit_pos[1])
        score = -dist_to_exit + (0.5 if new_pos in gold_items else 0)
        if score > best_score:
            best_score, best_action = score, move

    return [best_action, 0, 0]