def policy(env):
    # Strategy: Prioritize moving towards the exit while avoiding ghosts. Jump when necessary to reach higher platforms or avoid threats.
    # Use horizontal movement to align with exit, jump when exit is above or ghosts are near. Ignore unused a1/a2 actions.
    player = env.player
    exit_door = env.exit_door
    a0 = 0  # Default: no movement
    
    # Calculate horizontal direction to exit
    if player.centerx < exit_door.centerx:
        a0 = 4  # Move right
    elif player.centerx > exit_door.centerx:
        a0 = 3  # Move left
    
    # Jump if exit is above player and on ground, or to avoid nearby ghosts
    exit_above = exit_door.bottom < player.top
    ghost_near = any(abs(g['rect'].centerx - player.centerx) < 50 and abs(g['rect'].centery - player.centery) < 50 for g in env.ghosts)
    if env.on_ground and (exit_above or ghost_near):
        a0 = 1  # Jump
    
    return [a0, 0, 0]