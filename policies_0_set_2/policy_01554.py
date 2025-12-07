def policy(env):
    # Use BFS to find the shortest path to the exit, then take the first step.
    # This maximizes reward by minimizing moves (-1 per step) and reaching the exit quickly (+100).
    from collections import deque

    if env.robot_pos == env.exit_pos:
        return [0, 0, 0]

    parent = {}
    queue = deque([env.robot_pos])
    parent[env.robot_pos] = None
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]

    while queue:
        current = queue.popleft()
        if current == env.exit_pos:
            break
        x, y = current
        for dx, dy, act in directions:
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and
                next_pos not in parent and
                tuple(sorted((current, next_pos))) not in env.walls):
                parent[next_pos] = (current, act)
                queue.append(next_pos)

    if env.exit_pos not in parent:
        return [0, 0, 0]

    current = env.exit_pos
    while parent[current][0] != env.robot_pos:
        current = parent[current][0]
    action0 = parent[current][1]
    return [action0, 0, 0]