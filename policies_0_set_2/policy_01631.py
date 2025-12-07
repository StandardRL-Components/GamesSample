def policy(env):
    # Strategy: Prioritize matching parts to correct damage slots to maximize rewards (+10 per correct, -10 per wrong).
    # In SELECT_PART phase: Move cursor to first available part that matches any unrepaired damage, then select it.
    # In SELECT_LOCATION phase: Move cursor to first unrepaired damage matching held part, then place it.
    # Avoids errors by always targeting valid actions and breaks ties deterministically (left/right priority).
    
    if env.game_phase == "SELECT_PART":
        n_parts = len(env.available_parts)
        if n_parts == 0:
            return [0, 0, 0]  # No available parts
        
        # Identify required parts for unrepaired damages
        required_parts = set()
        for i, damage in enumerate(env.robot_config["damages"]):
            if not env.repaired_status[i]:
                required_parts.add(damage["part"])
        
        # Check current cursor part
        current_part = env.available_parts[env.part_cursor_idx]
        if current_part in required_parts:
            return [0, 1, 0]  # Select current part
        
        # Find nearest required part (prioritize right movement)
        for offset in range(1, n_parts):
            right_idx = (env.part_cursor_idx + offset) % n_parts
            if env.available_parts[right_idx] in required_parts:
                return [4, 0, 0]  # Move right
            left_idx = (env.part_cursor_idx - offset) % n_parts
            if env.available_parts[left_idx] in required_parts:
                return [3, 0, 0]  # Move left
        
        # No required parts available, select current part (will fail)
        return [0, 1, 0]
    
    elif env.game_phase == "SELECT_LOCATION":
        n_damages = len(env.robot_config["damages"])
        current_idx = env.location_cursor_idx
        held_part = env.held_part_key
        
        # Find target damage matching held part
        target_idx = None
        for i, damage in enumerate(env.robot_config["damages"]):
            if not env.repaired_status[i] and damage["part"] == held_part:
                target_idx = i
                break
        
        # If no match, target first unrepaired damage
        if target_idx is None:
            for i, repaired in enumerate(env.repaired_status):
                if not repaired:
                    target_idx = i
                    break
        
        if target_idx is None:
            return [0, 1, 0]  # All repaired (shouldn't happen)
        
        # Move toward target
        if current_idx == target_idx:
            return [0, 1, 0]  # Place part
        # Calculate shortest direction (prioritize right/left over up/down)
        right_dist = (target_idx - current_idx) % n_damages
        left_dist = (current_idx - target_idx) % n_damages
        if right_dist <= left_dist:
            return [4, 0, 0]  # Move right
        else:
            return [3, 0, 0]  # Move left
    
    return [0, 0, 0]  # Fallback