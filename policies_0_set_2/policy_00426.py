def policy(env):
    """
    Maximizes score by identifying and executing valid gem swaps that create matches.
    Uses a greedy approach: if a gem is selected, moves cursor to a valid adjacent swap position and activates swap.
    Otherwise, moves cursor to a gem that can form a valid swap and selects it. Avoids reshuffling due to high cost.
    """
    if env.game_state != "IDLE":
        return [0, 0, 0]
    
    board = env.board
    cursor = env.cursor_pos
    selected = env.selected_gem
    
    def is_valid_swap(pos1, pos2):
        if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) != 1:
            return False
        b = board.copy()
        x1, y1 = pos1
        x2, y2 = pos2
        b[y1, x1], b[y2, x2] = b[y2, x2], b[y1, x1]
        for y in range(8):
            for x in range(6):
                if b[y, x] != -1 and b[y, x] == b[y, x+1] == b[y, x+2]:
                    return True
        for x in range(8):
            for y in range(6):
                if b[y, x] != -1 and b[y, x] == b[y+1, x] == b[y+2, x]:
                    return True
        return False

    valid_swaps = []
    for y in range(8):
        for x in range(8):
            pos = (x, y)
            if x < 7 and is_valid_swap(pos, (x+1, y)):
                valid_swaps.append((pos, (x+1, y)))
            if y < 7 and is_valid_swap(pos, (x, y+1)):
                valid_swaps.append((pos, (x, y+1)))
    
    if not valid_swaps:
        return [0, 0, 0]
    
    if selected is not None:
        selected_tup = tuple(selected)
        valid_adjacents = []
        for swap in valid_swaps:
            if selected_tup == swap[0]:
                valid_adjacents.append(swap[1])
            elif selected_tup == swap[1]:
                valid_adjacents.append(swap[0])
        
        if not valid_adjacents:
            if tuple(cursor) != selected_tup:
                return [0, 1, 0]
            target_pos = valid_swaps[0][0]
            dx = target_pos[0] - cursor[0]
            dy = target_pos[1] - cursor[1]
            if dx > 0: return [4, 0, 0]
            if dx < 0: return [3, 0, 0]
            if dy > 0: return [2, 0, 0]
            return [1, 0, 0]
        
        if tuple(cursor) in valid_adjacents:
            return [0, 1, 0]
        target_pos = valid_adjacents[0]
        dx = target_pos[0] - cursor[0]
        dy = target_pos[1] - cursor[1]
        if dx > 0: return [4, 0, 0]
        if dx < 0: return [3, 0, 0]
        if dy > 0: return [2, 0, 0]
        return [1, 0, 0]
    
    valid_gems = set()
    for swap in valid_swaps:
        valid_gems.add(swap[0])
        valid_gems.add(swap[1])
    
    if tuple(cursor) in valid_gems:
        return [0, 1, 0]
    
    closest_gem = min(valid_gems, key=lambda g: abs(g[0]-cursor[0]) + abs(g[1]-cursor[1]))
    dx = closest_gem[0] - cursor[0]
    dy = closest_gem[1] - cursor[1]
    if dx > 0: return [4, 0, 0]
    if dx < 0: return [3, 0, 0]
    if dy > 0: return [2, 0, 0]
    return [1, 0, 0]