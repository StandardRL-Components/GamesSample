def policy(env):
    # Strategy: Prioritize moving green crystals to activate switches, then push any crystal to the exit.
    # Minimize Manhattan distance to targets while avoiding obstacles. Cycle selection only when necessary.
    exit_open = env.exit_open
    switches = env.switches
    crystals = env.crystals
    movable_crystals = env.movable_crystals
    selected_idx = env.selected_crystal_idx
    grid_w, grid_h = env.GRID_WIDTH, env.GRID_HEIGHT
    exit_pos = env.exit_pos

    if not movable_crystals:
        return [0, 0, 0]

    selected_global_idx = movable_crystals[selected_idx]
    selected_crystal = crystals[selected_global_idx]

    def slide_pos(pos, direction):
        dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][direction]
        x, y = pos
        obstacles = {c['pos'] for c in crystals if c['pos'] != selected_crystal['pos']}
        while True:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid_w and 0 <= ny < grid_h) or (nx, ny) in obstacles:
                break
            x, y = nx, ny
        return (x, y)

    if exit_open:
        if selected_crystal['pos'] == exit_pos:
            return [0, 0, 0]
        best_dir, best_dist = 0, float('inf')
        for direction in [1, 2, 3, 4]:
            new_pos = slide_pos(selected_crystal['pos'], direction)
            dist = abs(new_pos[0] - exit_pos[0]) + abs(new_pos[1] - exit_pos[1])
            if dist < best_dist:
                best_dist, best_dir = dist, direction
        if best_dir == 0 or slide_pos(selected_crystal['pos'], best_dir) == selected_crystal['pos']:
            return [0, 0, 0]
        return [best_dir, 0, 0]

    off_switches = [s for s in switches if not s['on']]
    if not off_switches:
        return [0, 0, 0]

    if selected_crystal['type'] != 'green':
        return [0, 1, 0]

    adj_switches = []
    for switch in off_switches:
        sx, sy = switch['pos']
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            adj = (sx + dx, sy + dy)
            if 0 <= adj[0] < grid_w and 0 <= adj[1] < grid_h:
                adj_switches.append(adj)

    if selected_crystal['pos'] in adj_switches:
        return [0, 1, 0]

    best_target = min(adj_switches, key=lambda p: abs(selected_crystal['pos'][0]-p[0]) + abs(selected_crystal['pos'][1]-p[1]))
    best_dir, best_dist = 0, float('inf')
    for direction in [1, 2, 3, 4]:
        new_pos = slide_pos(selected_crystal['pos'], direction)
        dist = abs(new_pos[0] - best_target[0]) + abs(new_pos[1] - best_target[1])
        if dist < best_dist:
            best_dist, best_dir = dist, direction
    if best_dir == 0 or slide_pos(selected_crystal['pos'], best_dir) == selected_crystal['pos']:
        return [0, 1, 0]
    return [best_dir, 0, 0]