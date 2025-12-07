def policy(env):
    """
    Navigates towards the exit while avoiding ghosts and walls. Uses Manhattan distance to prioritize
    movement directions that reduce distance to exit. Checks for wall collisions and ghost proximity
    to avoid penalties. Secondary actions are unused in this environment.
    """
    def rect_collision(r1, r2):
        return (r1[0] < r2[0] + r2[2] and r1[0] + r1[2] > r2[0] and
                r1[1] < r2[1] + r2[3] and r1[1] + r1[3] > r2[1])

    actions = [0, 1, 2, 3, 4]
    current_center = env.player_rect.center
    exit_center = env.exit_rect.center
    current_distance = abs(current_center[0] - exit_center[0]) + abs(current_center[1] - exit_center[1])
    
    best_action = 0
    best_score = -10**9
    
    for action in actions:
        dx, dy = 0, 0
        if action == 1:
            dy = -env.PLAYER_SPEED
        elif action == 2:
            dy = env.PLAYER_SPEED
        elif action == 3:
            dx = -env.PLAYER_SPEED
        elif action == 4:
            dx = env.PLAYER_SPEED
            
        new_x = env.player_rect.x + dx
        new_y = env.player_rect.y + dy
        new_rect = (new_x, new_y, env.player_rect.width, env.player_rect.height)
        
        if (new_x < 0 or new_y < 0 or 
            new_x + env.player_rect.width > env.SCREEN_WIDTH or 
            new_y + env.player_rect.height > env.SCREEN_HEIGHT):
            continue
            
        collision = False
        for wall in env.walls:
            if rect_collision(new_rect, (wall.x, wall.y, wall.width, wall.height)):
                collision = True
                break
        if collision:
            continue
            
        new_center = (new_x + env.player_rect.width/2, new_y + env.player_rect.height/2)
        new_distance = abs(new_center[0] - exit_center[0]) + abs(new_center[1] - exit_center[1])
        reduction = current_distance - new_distance
        
        min_ghost_dist_sq = float('inf')
        for ghost in env.ghosts:
            g_center = ghost.rect.center
            dist_sq = (new_center[0] - g_center[0])**2 + (new_center[1] - g_center[1])**2
            if dist_sq < min_ghost_dist_sq:
                min_ghost_dist_sq = dist_sq
                
        ghost_threshold = (2 * env.GHOST_SIZE) ** 2
        score = reduction
        if min_ghost_dist_sq < ghost_threshold:
            score -= 1000
            
        if score > best_score:
            best_score = score
            best_action = action
            
    return [best_action, 0, 0]