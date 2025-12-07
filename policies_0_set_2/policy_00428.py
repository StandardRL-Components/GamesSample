def policy(env):
    """Maximizes reward by building towers near the base and path to protect against enemies, prioritizing Cannon towers for range and damage when affordable, else Gatling. Moves cursor to optimal build locations."""
    if env.game_over:
        return [0, 0, 0]
    
    gold = env.gold
    cursor_x, cursor_y = env.cursor_grid_pos
    grid = env.tower_grid
    base_x, base_y = env.path_waypoints[-1]
    
    best_tile = None
    best_score = -float('inf')
    
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            tile = grid[x][y]
            if not tile['buildable'] or tile['occupied']:
                continue
                
            tile_x, tile_y = tile['pos']
            dist_base = ((tile_x - base_x)**2 + (tile_y - base_y)**2)**0.5
            min_path_dist = float('inf')
            
            for i in range(len(env.path_waypoints)-1):
                p1 = env.path_waypoints[i]
                p2 = env.path_waypoints[i+1]
                seg_vec = (p2[0]-p1[0], p2[1]-p1[1])
                seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2
                if seg_len_sq == 0:
                    dist = ((tile_x-p1[0])**2 + (tile_y-p1[1])**2)**0.5
                else:
                    t = max(0, min(1, ((tile_x-p1[0])*seg_vec[0] + (tile_y-p1[1])*seg_vec[1]) / seg_len_sq))
                    proj_x = p1[0] + t * seg_vec[0]
                    proj_y = p1[1] + t * seg_vec[1]
                    dist = ((tile_x-proj_x)**2 + (tile_y-proj_y)**2)**0.5
                if dist < min_path_dist:
                    min_path_dist = dist
                    
            score = 1.0/(dist_base + 1) + 1.0/(min_path_dist + 1)
            if score > best_score:
                best_score = score
                best_tile = (x, y)
    
    if best_tile is None:
        return [0, 0, 0]
        
    target_x, target_y = best_tile
    dx = target_x - cursor_x
    dy = target_y - cursor_y
    
    if dx == 0 and dy == 0:
        if gold >= 125:
            return [0, 0, 1]
        elif gold >= 75:
            return [0, 1, 0]
        else:
            return [0, 0, 0]
    elif dx > 0:
        return [4, 0, 0]
    elif dx < 0:
        return [3, 0, 0]
    elif dy > 0:
        return [2, 0, 0]
    else:
        return [1, 0, 0]