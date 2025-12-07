def policy(env):
    # Strategy: Prioritize attacking nearby enemies (immediate rewards), then move toward boss (level 3) or stairs (other levels).
    # Avoid unnecessary movement to minimize step penalty. Attack when enemy is in range (2.0 units).
    import math

    px, py = env.player['x'], env.player['y']
    current_pos = (px, py)

    # Check if any enemy is within attack range
    for enemy in env.enemies:
        dist = math.sqrt((enemy['x'] - px)**2 + (enemy['y'] - py)**2)
        if dist <= env.PLAYER_ATTACK_RANGE:
            return [0, 1, 0]  # Attack without moving

    # Determine target based on level and health
    if env.dungeon_level == 3 and any(e['type'] == 'boss' for e in env.enemies):
        # Target boss on level 3
        target = min(env.enemies, key=lambda e: (e['x'] - px)**2 + (e['y'] - py)**2)
        tx, ty = target['x'], target['y']
    elif env.player['health'] < 30 and env.dungeon_level < 3:
        # Low health: prioritize stairs to advance
        tx, ty = env.stairs_pos
    else:
        # Target nearest enemy or stairs if no enemies
        if env.enemies:
            target = min(env.enemies, key=lambda e: (e['x'] - px)**2 + (e['y'] - py)**2)
            tx, ty = target['x'], target['y']
        else:
            tx, ty = env.stairs_pos

    # Evaluate movement directions toward target
    best_action = 0  # Default: no movement
    best_dist = math.inf
    directions = [0, 1, 2, 3, 4]  # none, up, down, left, right
    for action in directions:
        dx, dy = 0, 0
        if action == 1: dx = -1
        elif action == 2: dx = 1
        elif action == 3: dy = -1
        elif action == 4: dy = 1

        nx, ny = px + dx, py + dy
        if not (0 <= nx < env.MAP_SIZE and 0 <= ny < env.MAP_SIZE):
            continue
        if env.grid[nx, ny] != 1:
            continue

        dist = (nx - tx)**2 + (ny - ty)**2
        if dist < best_dist:
            best_dist = dist
            best_action = action

    return [best_action, 0, 0]