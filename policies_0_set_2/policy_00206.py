def policy(env):
    # Strategy: Prioritize pushing crystals towards their targets when adjacent and beneficial.
    # If no immediate push available, move towards the nearest crystal not yet in target.
    # This reduces total distance efficiently and completes levels for high reward.
    def is_empty(pos):
        x, y = pos
        if not (0 <= x < env.grid_size[0] and 0 <= y < env.grid_size[1]):
            return False
        if env.grid[y, x] == 1:
            return False
        for crystal in env.crystals:
            if crystal['pos'] == (x, y):
                return False
        return True

    def manhattan_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    avatar_pos = env.avatar_pos
    move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

    # Check for beneficial pushes
    for crystal in env.crystals:
        if crystal['pos'] == crystal['target_pos']:
            continue
        cx, cy = crystal['pos']
        for move, (dx, dy) in move_map.items():
            push_pos = (cx + dx, cy + dy)
            avatar_req_pos = (cx - dx, cy - dy)
            if (avatar_pos == avatar_req_pos and 
                is_empty(push_pos) and 
                manhattan_dist(push_pos, crystal['target_pos']) < manhattan_dist(crystal['pos'], crystal['target_pos'])):
                return [move, 1, 0]

    # Move towards nearest incorrect crystal
    nearest_dist = float('inf')
    target_crystal = None
    for crystal in env.crystals:
        if crystal['pos'] != crystal['target_pos']:
            dist = manhattan_dist(avatar_pos, crystal['pos'])
            if dist < nearest_dist:
                nearest_dist = dist
                target_crystal = crystal

    if target_crystal:
        cx, cy = target_crystal['pos']
        dx = cx - avatar_pos[0]
        dy = cy - avatar_pos[1]
        if dx != 0:
            move = 4 if dx > 0 else 3
            new_pos = (avatar_pos[0] + move_map[move][0], avatar_pos[1] + move_map[move][1])
            if is_empty(new_pos):
                return [move, 0, 0]
        if dy != 0:
            move = 2 if dy > 0 else 1
            new_pos = (avatar_pos[0] + move_map[move][0], avatar_pos[1] + move_map[move][1])
            if is_empty(new_pos):
                return [move, 0, 0]

    # Default to no movement if no clear action
    return [0, 0, 0]