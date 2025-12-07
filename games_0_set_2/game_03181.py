import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set the SDL video driver to "dummy" for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to place selected tower, Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing defensive towers in an isometric world."
    )

    auto_advance = False

    # --- Helper Classes ---
    class Tower:
        def __init__(self, grid_pos, tower_type_info, iso_pos):
            self.grid_pos = grid_pos
            self.iso_pos = iso_pos
            self.type_info = tower_type_info
            self.cooldown = 0
            self.target = None

    class Enemy:
        def __init__(self, path, health, speed, value, color, size):
            self.path = path
            self.path_index = 0
            self.pixel_pos = list(path[0])
            self.max_health = health
            self.health = health
            self.speed = speed
            self.value = value
            self.color = color
            self.size = size
            self.is_alive = True
            
            # Add a slight random offset to path for visual variety
            self.offset_x = random.uniform(-3, 3)
            self.offset_y = random.uniform(-3, 3)

        def move(self):
            if not self.is_alive or self.path_index >= len(self.path) - 1:
                return

            target_pos = self.path[self.path_index + 1]
            dx = target_pos[0] - self.pixel_pos[0]
            dy = target_pos[1] - self.pixel_pos[1]
            dist = math.hypot(dx, dy)

            if dist < self.speed:
                self.path_index += 1
            else:
                self.pixel_pos[0] += (dx / dist) * self.speed
                self.pixel_pos[1] += (dy / dist) * self.speed

        def take_damage(self, amount):
            self.health -= amount
            if self.health <= 0:
                self.health = 0
                self.is_alive = False
                # sfx: enemy_explode

    class Projectile:
        def __init__(self, start_pos, target, damage, speed, color):
            self.pos = list(start_pos)
            self.target = target
            self.damage = damage
            self.speed = speed
            self.color = color
            self.is_active = True

        def move(self):
            if not self.is_active or not self.target.is_alive:
                self.is_active = False
                return

            dx = self.target.pixel_pos[0] + self.target.offset_x - self.pos[0]
            dy = self.target.pixel_pos[1] + self.target.offset_y - self.pos[1] - 10 # Aim slightly above center
            dist = math.hypot(dx, dy)

            if dist < self.speed:
                self.is_active = False
                self.target.take_damage(self.damage)
                # sfx: projectile_hit
            else:
                self.pos[0] += (dx / dist) * self.speed
                self.pos[1] += (dy / dist) * self.speed
    
    class Particle:
        def __init__(self, pos, color, start_size, duration, speed, angle):
            self.pos = list(pos)
            self.color = color
            self.size = start_size
            self.max_lifespan = duration
            self.lifespan = self.max_lifespan
            self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1
            return self.lifespan <= 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        # Set video mode to create a display surface, required for some surface operations
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game Constants & Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_BASE = (60, 180, 220)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_VALID_CURSOR = (80, 255, 80, 100)
        self.COLOR_INVALID_CURSOR = (255, 80, 80, 100)
        self.COLOR_HEALTH_GREEN = (40, 200, 40)
        self.COLOR_HEALTH_RED = (200, 40, 40)
        
        # --- Isometric Grid Setup ---
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_W, self.TILE_H = 40, 20
        self.ORIGIN_X, self.ORIGIN_Y = self.W // 2, 80

        # --- Game Mechanic Constants ---
        self.MAX_WAVES = 10
        self.MAX_STEPS = 2500
        self.WAVE_COOLDOWN = 150 # steps between waves
        self.MAX_BASE_HEALTH = 100
        self.STARTING_CURRENCY = 150

        self._define_path_and_grid()
        self._define_towers()
        
        self.shift_was_held = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # Optional, can be removed for submission

    def _define_path_and_grid(self):
        # Define path in grid coordinates
        path_grid_coords = [
            (-1, 4), (3, 4), (3, 1), (8, 1), (8, 7), (12, 7), (12, 4), (17, 4)
        ]
        # Convert to pixel coordinates
        self.enemy_path = [self._iso_to_screen(x, y) for x, y in path_grid_coords]
        self.base_pos_grid = (13, 4)
        self.base_pos_iso = self._iso_to_screen(*self.base_pos_grid)

        # Create a set of path tiles for quick lookup
        self.path_tiles = set()
        for i in range(len(path_grid_coords) - 1):
            p1 = path_grid_coords[i]
            p2 = path_grid_coords[i+1]
            if p1[0] == p2[0]: # Vertical line
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    self.path_tiles.add((p1[0], y))
            else: # Horizontal line
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    self.path_tiles.add((x, p1[1]))
        
        # Define valid placement spots
        self.placement_grid = []
        for y in range(self.GRID_H):
            row = []
            for x in range(self.GRID_W):
                if (x, y) not in self.path_tiles and (x, y) != self.base_pos_grid:
                    row.append((x, y))
            if row:
                self.placement_grid.append(row)
        
    def _define_towers(self):
        self.TOWER_TYPES = {
            0: {"name": "Gatling", "cost": 50, "dmg": 4, "range": 80, "rate": 5, "color": (200, 200, 0), "proj_speed": 8},
            1: {"name": "Cannon", "cost": 120, "dmg": 25, "range": 120, "rate": 25, "color": (255, 100, 0), "proj_speed": 6}
        }

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_W / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_H / 2)
        return screen_x, screen_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = self.MAX_BASE_HEALTH
        self.currency = self.STARTING_CURRENCY
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = self.WAVE_COOLDOWN
        self.wave_in_progress = False

        self.cursor_row = len(self.placement_grid) // 2
        self.cursor_col = len(self.placement_grid[self.cursor_row]) // 2
        self.selected_tower_type = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_btn, shift_btn = action
        reward = -0.01  # Small penalty for time passing

        # 1. Handle Player Input
        self._handle_input(movement, space_btn, shift_btn)
        
        # 2. Update Game Logic
        hit_reward = self._update_towers_and_projectiles()
        reward += hit_reward
        
        base_damage_reward, enemy_kill_reward = self._update_enemies()
        reward += base_damage_reward + enemy_kill_reward

        self._update_particles()
        
        wave_completion_reward = self._update_wave_manager()
        reward += wave_completion_reward

        # 3. Check for Termination
        terminated = False
        truncated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 50 # Losing penalty
        elif self.current_wave >= self.MAX_WAVES and not self.enemies and not self.wave_in_progress:
            self.game_over = True
            self.victory = True
            terminated = True
            reward += 100 # Winning bonus
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_btn, shift_btn):
        # --- Cursor Movement ---
        if movement == 1: # Up
            self.cursor_row = max(0, self.cursor_row - 1)
        elif movement == 2: # Down
            self.cursor_row = min(len(self.placement_grid) - 1, self.cursor_row + 1)
        
        # Adjust column index to be valid for the new row
        self.cursor_col = min(len(self.placement_grid[self.cursor_row]) - 1, self.cursor_col)

        if movement == 3: # Left
            self.cursor_col = max(0, self.cursor_col - 1)
        elif movement == 4: # Right
            self.cursor_col = min(len(self.placement_grid[self.cursor_row]) - 1, self.cursor_col + 1)

        # --- Tower Placement ---
        if space_btn == 1:
            grid_pos = self.placement_grid[self.cursor_row][self.cursor_col]
            tower_info = self.TOWER_TYPES[self.selected_tower_type]
            
            is_occupied = any(t.grid_pos == grid_pos for t in self.towers)
            
            if not is_occupied and self.currency >= tower_info["cost"]:
                self.currency -= tower_info["cost"]
                iso_pos = self._iso_to_screen(*grid_pos)
                self.towers.append(self.Tower(grid_pos, tower_info, iso_pos))
                # sfx: place_tower

        # --- Cycle Tower Type ---
        if shift_btn == 1 and not self.shift_was_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle
        self.shift_was_held = (shift_btn == 1)

    def _update_towers_and_projectiles(self):
        hit_reward = 0
        # Update towers
        for tower in self.towers:
            tower.cooldown = max(0, tower.cooldown - 1)
            if tower.cooldown == 0:
                # Find a target
                if not tower.target or not tower.target.is_alive or math.hypot(tower.iso_pos[0] - tower.target.pixel_pos[0], tower.iso_pos[1] - tower.target.pixel_pos[1]) > tower.type_info["range"]:
                    tower.target = None
                    closest_enemy = None
                    min_dist = float('inf')
                    for enemy in self.enemies:
                        dist = math.hypot(tower.iso_pos[0] - enemy.pixel_pos[0], tower.iso_pos[1] - enemy.pixel_pos[1])
                        if dist <= tower.type_info["range"]:
                            if dist < min_dist:
                                min_dist = dist
                                closest_enemy = enemy
                    tower.target = closest_enemy
                
                # Fire if target found
                if tower.target:
                    proj_start_pos = (tower.iso_pos[0], tower.iso_pos[1] - 10)
                    self.projectiles.append(self.Projectile(
                        proj_start_pos, tower.target, tower.type_info["dmg"], tower.type_info["proj_speed"], tower.type_info["color"]
                    ))
                    tower.cooldown = tower.type_info["rate"]
                    # sfx: tower_shoot

        # Update projectiles
        projectiles_to_keep = []
        for p in self.projectiles:
            p.move()
            if not p.is_active:
                if p.target and p.target.is_alive: # Hit confirmed
                    self._create_explosion(p.pos, p.color, 5)
                    hit_reward += 0.1
            else:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep
        return hit_reward

    def _update_enemies(self):
        base_damage_reward = 0
        enemy_kill_reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            if not enemy.is_alive:
                self.currency += enemy.value
                self.score += 10
                enemy_kill_reward += 1
                self._create_explosion(enemy.pixel_pos, enemy.color, 10)
                continue

            enemy.move()
            if enemy.path_index >= len(self.enemy_path) - 1:
                damage = int(enemy.health / 4) # Stronger enemies deal more damage
                self.base_health -= damage
                base_damage_reward -= damage
                # sfx: base_damage
            else:
                enemies_to_keep.append(enemy)
        self.enemies = enemies_to_keep
        return base_damage_reward, enemy_kill_reward

    def _update_wave_manager(self):
        wave_completion_reward = 0
        if not self.enemies and self.wave_in_progress:
            self.wave_in_progress = False
            self.wave_timer = self.WAVE_COOLDOWN
            self.score += 50
            wave_completion_reward += 10
            # sfx: wave_complete

        if not self.wave_in_progress and self.current_wave < self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self.wave_in_progress = True
                self._spawn_wave()
                # sfx: new_wave
        return wave_completion_reward

    def _spawn_wave(self):
        num_enemies = 3 + self.current_wave * 2
        base_health = 20 * (1.15 ** self.current_wave)
        base_speed = 0.8 * (1.05 ** self.current_wave)
        
        for i in range(num_enemies):
            # Stagger spawn positions slightly
            offset_path = [(p[0] - i * 10, p[1]) for p in self.enemy_path]
            health = base_health * random.uniform(0.9, 1.1)
            speed = base_speed * random.uniform(0.9, 1.1)
            self.enemies.append(self.Enemy(offset_path, health, speed, 10, (220, 50, 50), 6))

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            size = random.uniform(2, 5)
            duration = random.randint(10, 20)
            self.particles.append(self.Particle(pos, color, size, duration, speed, angle))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "currency": self.currency,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    def _render_game(self):
        # --- Draw Path and Grid ---
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                is_path = (x, y) in self.path_tiles
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                pygame.gfxdraw.line(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), color)
                pygame.gfxdraw.line(self.screen, int(p1[0]), int(p1[1]), int(p4[0]), int(p4[1]), color)

        # --- Draw Cursor and Tower Range ---
        cursor_grid_pos = self.placement_grid[self.cursor_row][self.cursor_col]
        cursor_iso_pos = self._iso_to_screen(*cursor_grid_pos)
        tower_info = self.TOWER_TYPES[self.selected_tower_type]
        
        is_occupied = any(t.grid_pos == cursor_grid_pos for t in self.towers)
        can_afford = self.currency >= tower_info["cost"]
        is_valid = not is_occupied and can_afford
        cursor_color = self.COLOR_VALID_CURSOR if is_valid else self.COLOR_INVALID_CURSOR
        
        # Create a new surface with per-pixel alpha for transparency
        range_surface = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, int(cursor_iso_pos[0]), int(cursor_iso_pos[1]), tower_info["range"], cursor_color)
        self.screen.blit(range_surface, (0, 0))
        
        # --- Sort and Draw Entities ---
        render_list = self.towers + self.enemies
        render_list.sort(key=lambda e: e.iso_pos[1] if isinstance(e, self.Tower) else e.pixel_pos[1])
        
        for entity in render_list:
            if isinstance(entity, self.Tower):
                pos = (int(entity.iso_pos[0]), int(entity.iso_pos[1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1]-5, 8, entity.type_info["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1]-5, 8, entity.type_info["color"])
            elif isinstance(entity, self.Enemy):
                pos = (int(entity.pixel_pos[0] + entity.offset_x), int(entity.pixel_pos[1] + entity.offset_y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1]-5, entity.size, entity.color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1]-5, entity.size, entity.color)
                # Health bar
                if entity.health < entity.max_health:
                    bar_w = 20
                    bar_h = 3
                    health_pct = entity.health / entity.max_health
                    pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos[0] - bar_w/2, pos[1] - 20, bar_w, bar_h))
                    pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0] - bar_w/2, pos[1] - 20, bar_w * health_pct, bar_h))
        
        # --- Draw Base ---
        p = self.base_pos_iso
        points = [(p[0], p[1]-15), (p[0]+15, p[1]-5), (p[0], p[1]+5), (p[0]-15, p[1]-5)]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BASE)
        # Base health bar
        bar_w, bar_h = 8, 50
        health_pct = self.base_health / self.MAX_BASE_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (p[0] - 30, p[1] - bar_h/2, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (p[0] - 30, p[1] - bar_h/2 + bar_h * (1-health_pct), bar_w, bar_h * health_pct))
        
        # --- Draw Projectiles & Particles ---
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, p.color)
        for p in self.particles:
            # gfxdraw does not support alpha in its color argument, so we skip it for particles
            # A more advanced implementation might use a separate surface for particles
            alpha = max(0, min(255, int(255 * (p.lifespan / p.max_lifespan))))
            color = p.color
            size = int(p.size * (p.lifespan / p.max_lifespan))
            if size > 0:
                # Use standard draw circle which is slower but supports alpha via surface alpha
                # For this fix, we'll just draw without alpha to keep it simple
                pygame.draw.circle(self.screen, color, (int(p.pos[0]), int(p.pos[1])), size)


    def _render_text(self, text, pos, font, color, center=False):
        img = font.render(text, True, color)
        rect = img.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(img, rect)

    def _render_ui(self):
        # Top-left UI
        self._render_text(f"SCORE: {self.score}", (10, 10), self.font_m, self.COLOR_TEXT)
        self._render_text(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", (10, 30), self.font_s, self.COLOR_TEXT)
        self._render_text(f"CREDITS: ${self.currency}", (10, 45), self.font_s, self.COLOR_TEXT)
        
        # Wave Timer
        if not self.wave_in_progress and self.current_wave < self.MAX_WAVES:
            secs = self.wave_timer / 30 # Assuming 30fps
            self._render_text(f"Next wave in: {secs:.1f}s", (self.W/2, 20), self.font_m, self.COLOR_TEXT, center=True)

        # Bottom-right UI (Selected Tower)
        tower_info = self.TOWER_TYPES[self.selected_tower_type]
        self._render_text(f"Selected: {tower_info['name']}", (self.W - 150, self.H - 65), self.font_m, tower_info['color'])
        self._render_text(f"Cost: ${tower_info['cost']}", (self.W - 150, self.H - 45), self.font_s, self.COLOR_TEXT)
        self._render_text(f"Dmg: {tower_info['dmg']} / Rng: {tower_info['range']}", (self.W - 150, self.H - 30), self.font_s, self.COLOR_TEXT)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.victory:
                self._render_text("VICTORY", (self.W/2, self.H/2 - 20), self.font_l, (100, 255, 100), center=True)
            else:
                self._render_text("GAME OVER", (self.W/2, self.H/2 - 20), self.font_l, (255, 100, 100), center=True)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will open a window, overriding the "dummy" video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    # The environment's self.screen is for headless rendering.
    # We create a new, visible screen for manual play.
    game_screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        if terminated or truncated:
            # After game over, wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
        else:
            # --- Get player input ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_btn = 1 if keys[pygame.K_SPACE] else 0
            shift_btn = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_btn, shift_btn]
            
            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
            
        # --- Render to screen ---
        # The 'obs' is the rendered frame from the headless environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()