def policy(env):
    # Prioritize moving towards the closest unfilled slot matching current crystal color to maximize immediate rewards (+10 per fill) and continuous rewards for moving closer. If no matching slot exists, target the closest unfilled slot to change color and continue progress.
    crystal_pos = env.crystal_pos
    crystal_color = env.crystal_color_idx
    unfilled_slots = [s for s in env.slots if not s['filled']]
    
    if not unfilled_slots:
        return [0, 0, 0]
    
    matching_slots = [s for s in unfilled_slots if s['color_idx'] == crystal_color]
    target_slot = min(matching_slots if matching_slots else unfilled_slots,
                     key=lambda s: env._get_manhattan_distance(crystal_pos, s['pos']))
    target_pos = target_slot['pos']
    
    best_action = 0
    best_dist = env._get_manhattan_distance(crystal_pos, target_pos)
    
    for action_idx, (dx, dy) in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)]):
        new_pos = (crystal_pos[0] + dx, crystal_pos[1] + dy)
        if 0 <= new_pos[0] < env.GRID_SIZE and 0 <= new_pos[1] < env.GRID_SIZE:
            new_dist = env._get_manhattan_distance(new_pos, target_pos)
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = action_idx
                
    return [best_action, 0, 0]