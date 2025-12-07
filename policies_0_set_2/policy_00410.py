def policy(env):
    """Policy for robot puzzle: Switch to uncharged robots, then move selected robot towards nearest unoccupied station (or nearest station if all occupied) using Manhattan distance and avoiding obstacles."""
    if env.game_over or all(r['on_station'] for r in env.robots):
        return [0, 0, 0]
    
    selected_robot_idx = env.selected_robot_idx
    current_robot = env.robots[selected_robot_idx]
    
    if current_robot['on_station'] and any(not r['on_station'] for r in env.robots) and not env.prev_shift_state:
        return [0, 0, 1]
    
    if not current_robot['on_station']:
        unoccupied_stations = []
        for s in env.stations:
            occupied = False
            for r in env.robots:
                if tuple(r['pos']) == tuple(s['pos']):
                    occupied = True
                    break
            if not occupied:
                unoccupied_stations.append(s)
                
        if unoccupied_stations:
            target_station = min(unoccupied_stations, key=lambda s: abs(s['pos'][0] - current_robot['pos'][0]) + abs(s['pos'][1] - current_robot['pos'][1]))
        else:
            target_station = min(env.stations, key=lambda s: abs(s['pos'][0] - current_robot['pos'][0]) + abs(s['pos'][1] - current_robot['pos'][1]))
            
        dx = target_station['pos'][0] - current_robot['pos'][0]
        dy = target_station['pos'][1] - current_robot['pos'][1]
        
        if abs(dx) > abs(dy):
            moves = [4 if dx > 0 else 3, 2 if dy > 0 else 1]
        else:
            moves = [2 if dy > 0 else 1, 4 if dx > 0 else 3]
        moves.extend([m for m in [1,2,3,4] if m not in moves])
        
        walls_set = set(tuple(w['pos']) for w in env.walls)
        other_robots_set = set(tuple(r['pos']) for i, r in enumerate(env.robots) if i != selected_robot_idx)
        
        for move in moves:
            new_x, new_y = current_robot['pos']
            if move == 1: new_y -= 1
            elif move == 2: new_y += 1
            elif move == 3: new_x -= 1
            elif move == 4: new_x += 1
            
            if new_x < 0 or new_x >= env.GRID_W or new_y < 0 or new_y >= env.GRID_H:
                continue
            if (new_x, new_y) in walls_set:
                continue
            if (new_x, new_y) in other_robots_set:
                continue
            return [move, 0, 0]
            
    return [0, 0, 0]