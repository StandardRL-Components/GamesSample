def policy(env):
    # Strategy: Maximize defense by prioritizing missile towers in optimal path-adjacent slots during build phases.
    # During waves, focus on eliminating enemies. If no affordable tower, sell weakest to upgrade.
    action = [0, 0, 0]
    tower_costs = [100, 250, 150, 75]
    slot_priority = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    if env.game_phase == "BUILD" or (env.game_phase == "WAVE" and env.money >= 250):
        affordable_tower = None
        if env.money >= 250:
            affordable_tower = 1
        elif env.money >= 100:
            affordable_tower = 0
        
        if affordable_tower is not None:
            empty_slot = None
            for slot in slot_priority:
                if slot not in env.towers:
                    empty_slot = slot
                    break
            
            if empty_slot is not None:
                if env.cursor_slot_index != empty_slot:
                    current = env.cursor_slot_index
                    forward = (empty_slot - current) % 12
                    backward = (current - empty_slot) % 12
                    action[0] = 2 if forward <= backward else 1
                elif env.selected_tower_type_index != affordable_tower:
                    current_type = env.selected_tower_type_index
                    forward_type = (affordable_tower - current_type) % 4
                    backward_type = (current_type - affordable_tower) % 4
                    action[0] = 4 if forward_type <= backward_type else 3
                else:
                    action[1] = 1
            else:
                for slot in reversed(slot_priority):
                    if slot in env.towers and env.towers[slot]["type"] != 1 and affordable_tower == 1:
                        if env.cursor_slot_index != slot:
                            current = env.cursor_slot_index
                            forward = (slot - current) % 12
                            backward = (current - slot) % 12
                            action[0] = 2 if forward <= backward else 1
                        else:
                            action[2] = 1
                        break
    
    return action