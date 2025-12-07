def policy(env):
    # Strategy: Prioritize connecting closest node pairs to minimize moves. If a node is selected, move directly to its pair. Otherwise, select the closest unconnected node. Avoid invalid actions and breaks ties by distance then left/right before up/down.
    cursor = env.cursor_pos
    selected = env.selected_node_info
    grid_w, grid_h = env.GRID_W, env.GRID_H
    
    def wrap_dist(a, b):
        dx = min(abs(a[0] - b[0]), grid_w - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), grid_h - abs(a[1] - b[1]))
        return dx + dy
    
    def move_toward(target):
        tx, ty = target
        cx, cy = cursor
        dx = (tx - cx + grid_w) % grid_w
        if dx > grid_w // 2:
            dx -= grid_w
        dy = (ty - cy + grid_h) % grid_h
        if dy > grid_h // 2:
            dy -= grid_h
        
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3
        else:
            return 2 if dy > 0 else 1

    unconnected = [pos for pos in env.nodes if not env._is_node_connected(pos)]
    if not unconnected:
        return [0, 0, 0]
    
    if selected:
        target_color = selected["color"]
        target_pos = None
        for pos in unconnected:
            if env.nodes[pos] == target_color and pos != selected["pos"]:
                target_pos = pos
                break
        if target_pos is None:
            return [0, 0, 1]
        if cursor == list(target_pos):
            return [0, 1, 0]
        return [move_toward(target_pos), 0, 0]
    else:
        closest = min(unconnected, key=lambda p: wrap_dist(cursor, p))
        if cursor == list(closest):
            return [0, 1, 0]
        return [move_toward(closest), 0, 0]