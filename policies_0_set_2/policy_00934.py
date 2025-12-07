def policy(env):
    # Use BFS to find the shortest path from player to goal, then return the first action.
    # This maximizes reward by minimizing battery usage and ensuring goal reachability.
    player_pos = env.player_pos
    goal_pos = env.goal_pos
    maze = env.maze

    if player_pos == goal_pos:
        return [0, 0, 0]

    queue = []
    start = (player_pos[0], player_pos[1])
    queue.append((start, []))
    visited = set()

    while queue:
        (y, x), path = queue.pop(0)
        if (y, x) in visited:
            continue
        visited.add((y, x))
        if [y, x] == goal_pos:
            if path:
                first_move = path[0]
                if first_move == (-1, 0):
                    return [1, 0, 0]
                elif first_move == (1, 0):
                    return [2, 0, 0]
                elif first_move == (0, -1):
                    return [3, 0, 0]
                elif first_move == (0, 1):
                    return [4, 0, 0]
            else:
                return [0, 0, 0]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] == 0:
                if (ny, nx) not in visited:
                    new_path = path + [(dy, dx)]
                    queue.append(((ny, nx), new_path))

    return [0, 0, 0]