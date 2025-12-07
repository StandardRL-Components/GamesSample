def policy(env):
    # Strategy: Avoid killer by hiding when too close, otherwise find nearest clue.
    # Maximizes reward by prioritizing safety (hiding) then clue collection (searching when near).
    safe_dist_sq = 2500  # 50^2
    search_radius_sq = 1600  # 40^2
    
    # Check killer proximity
    dx = env.player_pos[0] - env.killer_pos[0]
    dy = env.player_pos[1] - env.killer_pos[1]
    killer_dist_sq = dx*dx + dy*dy
    
    # Check hiding spot collision
    player_rect = {
        'left': env.player_pos[0] - env.PLAYER_SIZE[0]/2,
        'right': env.player_pos[0] + env.PLAYER_SIZE[0]/2,
        'top': env.player_pos[1] - env.PLAYER_SIZE[1],
        'bottom': env.player_pos[1]
    }
    in_hiding_spot = False
    for spot in env.hiding_spots:
        if (player_rect['left'] < spot.right and
            player_rect['right'] > spot.left and
            player_rect['top'] < spot.bottom and
            player_rect['bottom'] > spot.top):
            in_hiding_spot = True
            break

    # Escape killer if too close
    if killer_dist_sq < safe_dist_sq:
        if in_hiding_spot:
            return [0, 0, 1]  # Hide and stay still
        else:
            # Move away from killer
            if env.player_pos[0] < env.killer_pos[0]:
                return [3, 0, 0]  # Move left
            else:
                return [4, 0, 0]  # Move right

    # Find nearest unfound clue
    min_dist_sq = float('inf')
    target_clue = None
    for i, clue in enumerate(env.clue_locations):
        if not env.clues_found_mask[i]:
            dx = env.player_pos[0] - clue[0]
            dy = env.player_pos[1] - clue[1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                target_clue = clue

    if target_clue is None:
        return [0, 0, 0]  # No action if all clues found

    # Move toward clue and search if close
    move_dir = 4 if env.player_pos[0] < target_clue[0] else 3
    search_action = 1 if min_dist_sq < search_radius_sq else 0
    return [move_dir, search_action, 0]