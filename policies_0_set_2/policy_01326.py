def policy(env):
    # Strategy: Prioritize moving right to reach finish line quickly, while avoiding collisions with blocks.
    # Evaluate each movement action by projecting block positions and checking safety, then choose safest rightward move.
    # Always use a1=0 and a2=0 since secondary actions aren't used in this environment.
    actions = [0, 1, 2, 3, 4]  # none, up, down, left, right
    best_action = 4  # default to right
    best_score = -float('inf')
    player_size = env.PLAYER_SIZE
    player_rect = pygame.Rect(env.player_pos.x - player_size/2, env.player_pos.y - player_size/2, player_size, player_size)
    
    for action in actions:
        if action == 0:
            move_vec = (0, 0)
        elif action == 1:
            move_vec = (0, -env.PLAYER_SPEED)
        elif action == 2:
            move_vec = (0, env.PLAYER_SPEED)
        elif action == 3:
            move_vec = (-env.PLAYER_SPEED, 0)
        else:
            move_vec = (env.PLAYER_SPEED, 0)
            
        new_x = max(player_size, min(env.player_pos.x + move_vec[0], env.WIDTH - player_size))
        new_y = max(player_size, min(env.player_pos.y + move_vec[1], env.HEIGHT - player_size))
        new_rect = pygame.Rect(new_x - player_size/2, new_y - player_size/2, player_size, player_size)
        
        collision = False
        for block in env.blocks:
            block_rect = block['rect'].copy()
            block_rect.y += block['speed']
            if block_rect.colliderect(new_rect):
                collision = True
                break
                
        if not collision:
            score = new_x  # prioritize rightward movement
            if score > best_score:
                best_score = score
                best_action = action
                
    return [best_action, 0, 0]