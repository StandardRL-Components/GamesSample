def policy(env):
    # Strategy: Prioritize placing firebreaks on unburnt tiles adjacent to fire and closest to the grid edge to contain spread.
    # Move cursor to the most critical tile (min distance to edge) if not already there, then place firebreak.
    grid = env.grid
    cursor = env.cursor_pos
    size = env.GRID_SIZE

    def danger(pos):
        x, y = pos
        return min(x, size-1-x, y, size-1-y)

    def is_critical(x, y):
        if grid[y, x] != env.TILE_UNBURNT:
            return False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny, nx] == env.TILE_FIRE:
                return True
        return False

    critical_tiles = []
    for x in range(size):
        for y in range(size):
            if is_critical(x, y):
                critical_tiles.append((x,y))
                
    if critical_tiles:
        target = min(critical_tiles, key=danger)
        current_x, current_y = cursor
        current_critical = is_critical(current_x, current_y)
        current_danger = danger(cursor) if current_critical else float('inf')
        target_danger = danger(target)
        
        if current_critical and current_danger <= target_danger:
            return [0, 1, 0]
        else:
            dx = target[0] - current_x
            dy = target[1] - current_y
            if abs(dx) > abs(dy):
                if dx > 0:
                    return [4, 0, 0]
                else:
                    return [3, 0, 0]
            else:
                if dy > 0:
                    return [2, 0, 0]
                else:
                    return [1, 0, 0]
    else:
        return [0, 0, 0]