def policy(env):
    # Strategy: Prioritize pushing unlit crystals onto unlit plates for immediate reward (+5 per plate, -0.2 move cost).
    # If no direct push available, move unlit crystals closer to unlit plates to set up future rewards.
    # Avoid moving lit crystals to prevent unlighting plates. Change selection efficiently when needed.
    def is_plate(pos):
        for p in env.plates:
            if p[0] == pos[0] and p[1] == pos[1]:
                return True
        return False

    def simulate_push(crystal_idx, direction):
        dxdy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = dxdy[direction]
        crystal_pos = env.crystals[crystal_idx]
        current = crystal_pos.copy()
        while True:
            next_pos = [current[0] + dx, current[1] + dy]
            if not (0 <= next_pos[0] < env.GRID_WIDTH and 0 <= next_pos[1] < env.GRID_HEIGHT):
                break
            collision = False
            for i, c in enumerate(env.crystals):
                if i != crystal_idx and c[0] == next_pos[0] and c[1] == next_pos[1]:
                    collision = True
                    break
            if collision:
                break
            current = next_pos
        return current

    if env.game_over:
        return [0, 0, 0]

    unlit_plates = []
    for i, plate in enumerate(env.plates):
        if not env.lit_mask[i]:
            unlit_plates.append(plate)

    if not unlit_plates:
        return [0, 0, 0]

    best_direct = None
    for crystal_idx in range(env.NUM_CRYSTALS):
        if is_plate(env.crystals[crystal_idx]):
            continue
        for direction in [1, 2, 3, 4]:
            new_pos = simulate_push(crystal_idx, direction)
            if new_pos in unlit_plates:
                best_direct = (crystal_idx, direction)
                break
        if best_direct is not None:
            break

    if best_direct is not None:
        crystal_idx, direction = best_direct
        if env.selected_crystal_idx == crystal_idx:
            return [direction, 1, 0]
        else:
            n = env.NUM_CRYSTALS
            current = env.selected_crystal_idx
            forward = (crystal_idx - current) % n
            backward = (current - crystal_idx) % n
            if forward <= backward:
                return [1, 0, 0]
            else:
                return [2, 0, 0]

    best_dist = float('inf')
    best_push = None
    for crystal_idx in range(env.NUM_CRYSTALS):
        if is_plate(env.crystals[crystal_idx]):
            continue
        for direction in [1, 2, 3, 4]:
            new_pos = simulate_push(crystal_idx, direction)
            min_dist = min(abs(new_pos[0] - p[0]) + abs(new_pos[1] - p[1]) for p in unlit_plates)
            if min_dist < best_dist:
                best_dist = min_dist
                best_push = (crystal_idx, direction)

    if best_push is None:
        return [0, 0, 0]

    crystal_idx, direction = best_push
    if env.selected_crystal_idx == crystal_idx:
        return [direction, 1, 0]
    else:
        n = env.NUM_CRYSTALS
        current = env.selected_crystal_idx
        forward = (crystal_idx - current) % n
        backward = (current - crystal_idx) % n
        if forward <= backward:
            return [1, 0, 0]
        else:
            return [2, 0, 0]