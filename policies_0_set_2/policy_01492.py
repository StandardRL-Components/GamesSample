def policy(env):
    # Prioritize attacking adjacent enemies to eliminate threats and gain rewards, then move towards gold/stairs while avoiding danger.
    # Strategy: Attack if enemy adjacent and safe (normal enemy or boss when health allows), else move to highest-scoring cell (gold > stairs > safe tile).
    player_x, player_y = env.player.x, env.player.y
    
    # Check for adjacent enemies and attack conditions
    adjacent_enemy = False
    adjacent_boss = False
    boss_killable = env.boss and env.boss.health <= env.PLAYER_ATTACK_DAMAGE
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        nx, ny = player_x + dx, player_y + dy
        for enemy in env.enemies:
            if (enemy.x, enemy.y) == (nx, ny):
                adjacent_enemy = True
                break
        if env.boss and (env.boss.x, env.boss.y) == (nx, ny):
            adjacent_boss = True
            break
    
    # Attack if safe: always for normal enemies, or for boss when health allows or killable
    if adjacent_enemy or (adjacent_boss and (env.player.health > env.BOSS_ATTACK_DAMAGE or boss_killable)):
        return [0, 1, 0]
    
    # Evaluate movement directions (including no-op)
    best_score = -9999
    best_move = 0
    directions = [0, 1, 2, 3, 4]  # none, up, down, left, right
    for move in directions:
        if move == 0:
            nx, ny = player_x, player_y
        else:
            dx, dy = {1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}[move]
            nx, ny = player_x + dx, player_y + dy
            if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT) or env.dungeon_map[nx, ny] != 1:
                continue
        
        score = 0
        # Gold collection bonus
        if (nx, ny) in env.gold_pieces:
            score += 1.0
        # Stairs proximity bonus
        stairs_dist = abs(nx - env.stairs_pos[0]) + abs(ny - env.stairs_pos[1])
        player_dist = abs(player_x - env.stairs_pos[0]) + abs(player_y - env.stairs_pos[1])
        if stairs_dist < player_dist:
            score += 0.1
        # Enemy proximity penalty
        for enemy in env.enemies:
            if abs(nx - enemy.x) + abs(ny - enemy.y) == 1:
                score -= 1.0
        if env.boss and env.boss.health > 0 and abs(nx - env.boss.x) + abs(ny - env.boss.y) == 1:
            score -= 2.0
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return [best_move, 0, 0]