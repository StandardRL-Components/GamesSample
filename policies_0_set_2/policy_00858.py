def policy(env):
    # Strategy: Jump towards goal asteroid with adaptive direction and jump strength.
    # When grounded, compute optimal jump direction toward goal using vector math.
    # Use long jumps for distant targets, short jumps for precision when close.
    # Avoid left movements since goal is always rightwards. No action when airborne.
    if not env.is_grounded:
        return [0, 0, 0]
    
    goal_pos = env.asteroids[env.goal_asteroid_idx]['pos']
    player_pos = env.player_pos
    vector_to_goal = goal_pos - player_pos
    distance = vector_to_goal.length()
    
    if abs(vector_to_goal.y) < abs(vector_to_goal.x):
        movement = 4  # Prioritize horizontal right movement
    else:
        movement = 2 if vector_to_goal.y > 0 else 1  # Vertical adjustment if needed
    
    jump_type = 1 if distance > 150 else 0  # Long jump for far targets
    return [movement, jump_type, 0]