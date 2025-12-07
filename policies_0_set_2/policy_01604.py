def policy(env):
    # Strategy: Prioritize charging robots by moving to nearest valid station using Manhattan distance.
    # In SELECT phase, choose the uncharged robot closest to any free station.
    # In MOVE phase, move selected robot toward closest free station via valid adjacent moves.
    # If no immediate progress possible, cancel selection to avoid wasting moves.
    if env.game_phase == "SELECT":
        uncharged = [r for r in env.robots if not r['charged']]
        if not uncharged:
            return [0, 0, 0]
        free_stations = [s['pos'] for s in env.stations if not any(r['pos'] == s['pos'] and r['charged'] for r in env.robots)]
        best_robot = min(uncharged, key=lambda r: min(
            abs(r['pos'][0]-s[0]) + abs(r['pos'][1]-s[1]) for s in free_stations
        ) if free_stations else uncharged[0])
        if env.cursor_pos == best_robot['pos']:
            return [0, 1, 0]
        dx = best_robot['pos'][0] - env.cursor_pos[0]
        dy = best_robot['pos'][1] - env.cursor_pos[1]
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    else:
        robot = env.robots[env.selected_robot_idx]
        free_stations = [s['pos'] for s in env.stations if not any(r['pos'] == s['pos'] and r['charged'] for r in env.robots)]
        if not free_stations:
            return [0, 0, 1]
        target = min(free_stations, key=lambda s: abs(s[0]-robot['pos'][0]) + abs(s[1]-robot['pos'][1]))
        dx = target[0] - robot['pos'][0]
        dy = target[1] - robot['pos'][1]
        moves = []
        if dx != 0:
            moves.append(4 if dx > 0 else 3)
        if dy != 0:
            moves.append(2 if dy > 0 else 1)
        for move in moves:
            new_pos = list(robot['pos'])
            if move == 1: new_pos[1] -= 1
            elif move == 2: new_pos[1] += 1
            elif move == 3: new_pos[0] -= 1
            elif move == 4: new_pos[0] += 1
            if env._is_valid_move(robot, new_pos):
                if env.cursor_pos == new_pos:
                    return [0, 1, 0]
                dx_c = new_pos[0] - env.cursor_pos[0]
                dy_c = new_pos[1] - env.cursor_pos[1]
                if abs(dx_c) > abs(dy_c):
                    return [4 if dx_c > 0 else 3, 0, 0]
                else:
                    return [2 if dy_c > 0 else 1, 0, 0]
        return [0, 0, 1]