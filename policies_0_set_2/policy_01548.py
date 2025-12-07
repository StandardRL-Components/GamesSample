def policy(env):
    # This policy maximizes reward by focusing on rotating crystals in the beam path to illuminate targets.
    # It simulates clockwise and counterclockwise rotations for the currently selected crystal and chooses the one that lights the most targets.
    # If no improvement, it moves selection to the next crystal in the beam path to continue searching for optimal rotations.
    
    def simulate_beam(crystal_rotations):
        beam_pos = env.light_source['grid_pos']
        beam_dir = env.light_source['direction']
        lit_targets = set()
        for _ in range(env.GRID_WIDTH + env.GRID_HEIGHT):
            next_pos = env._get_next_pos(beam_pos, beam_dir)
            if next_pos in target_positions:
                lit_targets.add(next_pos)
            beam_pos = next_pos
            crystal_hit = None
            for idx, pos in enumerate(crystal_positions):
                if pos == beam_pos:
                    crystal_hit = idx
                    break
            if crystal_hit is not None:
                beam_dir = env.REFLECTIONS[crystal_rotations[crystal_hit]][beam_dir]
            if not (0 <= beam_pos[0] < env.GRID_WIDTH and 0 <= beam_pos[1] < env.GRID_HEIGHT):
                break
        return len(lit_targets)

    crystal_positions = [c['grid_pos'] for c in env.rotatable_crystals]
    target_positions = [t['grid_pos'] for t in env.targets]
    current_rotations = [c['rotation'] for c in env.rotatable_crystals]
    current_lit = sum(1 for t in env.targets if t['is_lit'])
    
    current_crystal_pos = env.rotatable_crystals[env.selected_crystal_idx]['grid_pos']
    beam_path_crystals = set()
    beam_pos = env.light_source['grid_pos']
    beam_dir = env.light_source['direction']
    for _ in range(env.GRID_WIDTH + env.GRID_HEIGHT):
        next_pos = env._get_next_pos(beam_pos, beam_dir)
        if next_pos in crystal_positions:
            beam_path_crystals.add(next_pos)
        beam_pos = next_pos
        crystal_hit = None
        for idx, pos in enumerate(crystal_positions):
            if pos == beam_pos:
                crystal_hit = idx
                break
        if crystal_hit is not None:
            beam_dir = env.REFLECTIONS[current_rotations[crystal_hit]][beam_dir]
        if not (0 <= beam_pos[0] < env.GRID_WIDTH and 0 <= beam_pos[1] < env.GRID_HEIGHT):
            break

    if current_crystal_pos in beam_path_crystals:
        new_rotations_cw = current_rotations[:]
        new_rotations_cw[env.selected_crystal_idx] = (new_rotations_cw[env.selected_crystal_idx] + 1) % 3
        lit_cw = simulate_beam(new_rotations_cw)
        new_rotations_ccw = current_rotations[:]
        new_rotations_ccw[env.selected_crystal_idx] = (new_rotations_ccw[env.selected_crystal_idx] - 1) % 3
        lit_ccw = simulate_beam(new_rotations_ccw)
        if lit_cw > current_lit or lit_ccw > current_lit:
            if lit_cw >= lit_ccw:
                return [0, 1, 0]
            else:
                return [0, 0, 1]
    
    num_crystals = len(env.rotatable_crystals)
    for i in range(1, num_crystals):
        next_idx = (env.selected_crystal_idx + i) % num_crystals
        if env.rotatable_crystals[next_idx]['grid_pos'] in beam_path_crystals:
            if i <= num_crystals // 2:
                return [2, 0, 0]
            else:
                return [1, 0, 0]
    
    return [2, 0, 0]