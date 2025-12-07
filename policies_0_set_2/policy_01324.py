def policy(env):
    """
    Survival strategy: Avoid active spikes and high-risk inactive spikes by moving to safest adjacent position.
    Prioritize maximizing distance from active spikes and minimizing risk from near-activation spikes.
    Secondary actions unused in this environment, so set to 0.
    """
    # Get current player position and spike states
    player_pos = env.player_pos
    spikes = env.spikes
    
    # Evaluate danger for each candidate movement (including no-op)
    actions = [0, 1, 2, 3, 4]  # none, up, down, left, right
    best_action = 0
    min_danger = float('inf')
    
    for a in actions:
        # Calculate candidate position after movement
        cand_pos = player_pos.copy()
        if a == 1:  # up
            cand_pos[1] -= env.PLAYER_SPEED
        elif a == 2:  # down
            cand_pos[1] += env.PLAYER_SPEED
        elif a == 3:  # left
            cand_pos[0] -= env.PLAYER_SPEED
        elif a == 4:  # right
            cand_pos[0] += env.PLAYER_SPEED
        
        # Clamp to screen bounds
        cand_pos[0] = max(env.PLAYER_RADIUS, min(env.WIDTH - env.PLAYER_RADIUS, cand_pos[0]))
        cand_pos[1] = max(env.PLAYER_RADIUS, min(env.HEIGHT - env.PLAYER_RADIUS, cand_pos[1]))
        
        # Calculate danger score for this position
        danger = 0
        for spike in spikes:
            dist = math.hypot(cand_pos[0] - spike['pos'][0], cand_pos[1] - spike['pos'][1])
            if spike['state'] == 'active':
                # Active spikes are immediate threat - prioritize distance
                danger += 1000 / (dist + 1e-5)
            else:
                # Inactive spikes with low timer are high risk
                activation_risk = max(0, 1 - spike['activation_timer'] / 60)  # Normalize risk
                danger += 100 * activation_risk / (dist + 1e-5)
        
        # Prefer current position if dangers are equal to avoid unnecessary movement
        if danger < min_danger or (danger == min_danger and a == 0):
            min_danger = danger
            best_action = a
    
    return [best_action, 0, 0]