def policy(env):
    # Navigate towards the nearest part using BFS to find the shortest path, avoiding walls.
    # This minimizes moves and maximizes part collection efficiency, balancing immediate rewards with future opportunities.
    from collections import deque

    parts = set(env.parts)
    if not parts:
        return [0, 0, 0]

    start = env.robot_pos
    if start in parts:
        return [0, 0, 0]

    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    visited = set()
    queue = deque()
    queue.append((start, None))
    visited.add(start)
    parent = {}

    while queue:
        current, first_action = queue.popleft()
        if current in parts:
            return [first_action, 0, 0] if first_action is not None else [0, 0, 0]

        x, y = current
        for dx, dy, a in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and
                env.grid[ny, nx] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                action_to_use = first_action if first_action is not None else a
                queue.append(((nx, ny), action_to_use))
                parent[(nx, ny)] = (current, action_to_use)

    for dx, dy, a in directions:
        nx, ny = start[0] + dx, start[1] + dy
        if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and
            env.grid[ny, nx] == 0):
            return [a, 0, 0]
    return [0, 0, 0]