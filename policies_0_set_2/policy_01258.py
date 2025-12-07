def policy(env):
    # Strategy: Avoid projectiles by moving away from nearest threats while minimizing stationary penalty.
    # Scan for red projectiles and move perpendicular to the closest one's velocity vector to dodge effectively.
    # If no immediate threat, move in a circular pattern to avoid center (projectile target zone) and edges (spawn points).
    
    # Get current screen surface from environment (read-only)
    screen = env.screen
    width, height = screen.get_width(), screen.get_height()
    
    # Find player position by scanning for green player color
    player_pos = None
    for x in range(0, width, 5):  # Downsample for efficiency
        for y in range(0, height, 5):
            color = screen.get_at((x, y))
            if 40 <= color[0] <= 60 and 245 <= color[1] <= 255 and 140 <= color[2] <= 160:  # Player green
                player_pos = (x, y)
                break
        if player_pos:
            break
    
    if not player_pos:
        return [0, 0, 0]  # Default to no movement if player not found
    
    # Find nearest projectile by scanning for red color
    nearest_proj = None
    min_dist = float('inf')
    for x in range(0, width, 8):  # Downsample more for projectiles
        for y in range(0, height, 8):
            color = screen.get_at((x, y))
            if 245 <= color[0] <= 255 and 70 <= color[1] <= 90 and 70 <= color[2] <= 90:  # Projectile red
                dist = (x - player_pos[0])**2 + (y - player_pos[1])**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_proj = (x, y)
    
    # Move based on threat assessment
    if nearest_proj:
        # Calculate vector from player to projectile
        dx = nearest_proj[0] - player_pos[0]
        dy = nearest_proj[1] - player_pos[1]
        
        # Move perpendicular to threat vector (dodge)
        if abs(dx) > abs(dy):
            movement = 2 if dy > 0 else 1  # Vertical dodge
        else:
            movement = 4 if dx > 0 else 3  # Horizontal dodge
    else:
        # Circular patrol pattern when no immediate threats
        center_x, center_y = width // 2, height // 2
        rel_x = player_pos[0] - center_x
        rel_y = player_pos[1] - center_y
        
        # Move tangentially to center
        if abs(rel_x) > abs(rel_y):
            movement = 2 if rel_y < 0 else 1  # Vertical movement
        else:
            movement = 4 if rel_x < 0 else 3  # Horizontal movement
    
    return [movement, 0, 0]  # a1 and a2 unused in this environment