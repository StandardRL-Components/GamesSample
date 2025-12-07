def policy(env):
    # Strategy: Prioritize building Gatling towers early for cost efficiency, then Cannons for stronger waves.
    # Focus on high-value spots near path corners for maximum coverage. Move cursor systematically to empty spots.
    # Only change tower type when necessary and place when affordable on valid spots to avoid wasting actions.
    desired_tower_type = 0
    if env.wave_number >= 4 and env.money >= 150:
        desired_tower_type = 1
    if env.selected_tower_type_idx != desired_tower_type:
        return [0, 0, 1]
    
    priority_spots = [7, 19, 22, 10, 6, 8, 18, 20, 11, 9, 23, 21]
    target_spot = env.cursor_index
    for spot in priority_spots:
        pos = env.PLACEMENT_SPOTS[spot]
        if not any(t['pos'] == pos for t in env.towers):
            target_spot = spot
            break
            
    cur_col, cur_row = env.cursor_index % 6, env.cursor_index // 6
    tgt_col, tgt_row = target_spot % 6, target_spot // 6
    a0 = 0
    if tgt_col > cur_col:
        a0 = 4
    elif tgt_col < cur_col:
        a0 = 3
    elif tgt_row > cur_row:
        a0 = 2
    elif tgt_row < cur_row:
        a0 = 1
        
    a1 = 0
    if env.cursor_index == target_spot:
        spec = env.TOWER_SPECS[env.selected_tower_type_idx]
        pos = env.PLACEMENT_SPOTS[env.cursor_index]
        if not any(t['pos'] == pos for t in env.towers) and env.money >= spec['cost']:
            a1 = 1
            
    return [a0, a1, 0]