
# Generated: 2025-08-27T18:17:49.776205
# Source Brief: brief_01788.md
# Brief Index: 1788

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Unit:
    def __init__(self, grid_pos, cost):
        self.grid_pos = grid_pos
        self.attack_range = 80
        self.attack_cooldown = 0
        self.attack_speed = 30  # frames per attack
        self.cost = cost

class Enemy:
    def __init__(self, path, health, speed, size, value):
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = size
        self.value = value # resources gained on kill

class Projectile:
    def __init__(self, start_pos, target_enemy, speed=15):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.speed = speed

class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the deployment cursor. Press space to deploy a turret at the cursor's location."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically deploying turrets in this isometric tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    OFFSET_X, OFFSET_Y = SCREEN_WIDTH // 2, 80

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_PATH = (60, 70, 100)
    COLOR_BASE = (0, 120, 255)
    COLOR_BASE_SHADOW = (0, 60, 130)
    COLOR_UNIT = (0, 255, 120)
    COLOR_UNIT_SHADOW = (0, 130, 60)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_SHADOW = (130, 25, 25)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_UI_BG = (10, 15, 30, 200)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR_FG = (50, 200, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self._define_path()
        self.reset()
        self.validate_implementation()

    def _define_path(self):
        self.path_grid_coords = [
            (-1, 4), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (4, 1),
            (5, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 6), (8, 6),
            (9, 6), (10, 6), (11, 6), (11, 7), (11, 8), (12, 8), (13, 8), (14, 8),
            (15, 8), (16, 8)
        ]
        self.path_screen_coords = [self._iso_to_screen(x, y) for x, y in self.path_grid_coords]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 80
        self.unit_cost = 40
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        
        self.enemies = []
        self.units = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_countdown = 150 # frames before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemy_spawn_timer = 0
        
        self.path_grid_set = set(self.path_grid_coords)
        self.base_pos_grid = (15, 4)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.001 # Small penalty for existing
        
        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            self._cleanup_objects()
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.game_over = True

        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Deploy unit
        is_valid_tile = tuple(self.cursor_pos) not in self.path_grid_set and \
                        tuple(self.cursor_pos) != self.base_pos_grid
        
        if space_held and not self.last_space_held and self.resources >= self.unit_cost and is_valid_tile:
            # Check if a unit is already on this tile
            if not any(u.grid_pos == self.cursor_pos for u in self.units):
                self.units.append(Unit(self.cursor_pos.copy(), self.unit_cost))
                self.resources -= self.unit_cost
                # sfx: deploy_unit
                self._create_particles(self._iso_to_screen(*self.cursor_pos), 20, self.COLOR_UNIT, 2)
        
        self.last_space_held = bool(space_held)

    def _update_game_state(self):
        reward = 0
        
        # Wave management
        if self.wave_countdown > 0:
            self.wave_countdown -= 1
        elif not self.enemies and self.enemies_spawned == self.enemies_in_wave:
            if self.current_wave < 10:
                self._start_next_wave()
            
        # Enemy spawning
        if self.enemies_spawned < self.enemies_in_wave and self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemies_spawned += 1
            self.enemy_spawn_timer = 45 # frames between spawns
        self.enemy_spawn_timer -= 1
        
        # Update units
        for unit in self.units:
            unit.attack_cooldown -= 1
            if unit.attack_cooldown <= 0:
                target = self._find_closest_enemy(unit)
                if target:
                    # sfx: shoot_projectile
                    start_pos = self._iso_to_screen(*unit.grid_pos)
                    self.projectiles.append(Projectile(start_pos, target))
                    unit.attack_cooldown = unit.attack_speed

        # Update projectiles
        for p in self.projectiles:
            if p.target.health <= 0: # Target already dead
                p.pos = [-100, -100] # Mark for removal
                continue
            
            target_pos = self._iso_to_screen(*p.target.pos)
            direction = np.array(target_pos) - np.array(p.pos)
            dist = np.linalg.norm(direction)
            if dist < p.speed:
                p.target.health -= 10
                reward += 0.1 # Hit reward
                # sfx: hit_enemy
                self._create_particles(target_pos, 10, self.COLOR_PROJECTILE, 1)
                p.pos = [-100, -100] # Mark for removal
                if p.target.health <= 0:
                    reward += 1.0 # Kill reward
                    self.resources += p.target.value
                    self._create_particles(target_pos, 30, self.COLOR_ENEMY, 3)
            else:
                p.pos += (direction / dist) * p.speed

        # Update enemies
        for enemy in self.enemies:
            if enemy.path_index < len(self.path_screen_coords) - 1:
                target_pos = self.path_screen_coords[enemy.path_index + 1]
                direction = np.array(target_pos) - np.array(enemy.pos)
                dist = np.linalg.norm(direction)
                if dist < enemy.speed:
                    enemy.path_index += 1
                    enemy.pos = list(self.path_screen_coords[enemy.path_index])
                else:
                    enemy.pos += (direction / dist) * enemy.speed
            else: # Reached base
                self.base_health -= 10
                reward -= 10 # Base damage penalty
                # sfx: base_damage
                self._create_particles(self._iso_to_screen(*self.base_pos_grid), 40, self.COLOR_BASE, 4)
                enemy.health = 0 # Mark for removal

        # Update particles
        for particle in self.particles:
            particle.pos[0] += particle.vel[0]
            particle.pos[1] += particle.vel[1]
            particle.lifetime -= 1
            particle.size = max(0, particle.size - 0.1)

        return reward

    def _cleanup_objects(self):
        self.enemies = [e for e in self.enemies if e.health > 0]
        self.projectiles = [p for p in self.projectiles if p.pos[0] > -100]
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.win = False
            return True
        if self.current_wave == 10 and not self.enemies and self.enemies_spawned == self.enemies_in_wave:
            self.win = True
            return True
        if self.steps >= 2500: # Max steps
            self.win = False
            return True
        return False
        
    def _start_next_wave(self):
        self.current_wave += 1
        self.enemies_in_wave = 5 + self.current_wave * 2
        self.enemies_spawned = 0
        self.wave_countdown = 120 # time between waves

    def _spawn_enemy(self):
        health = 20 * (1.1 ** (self.current_wave - 1))
        speed = 1.0 * (1.05 ** (self.current_wave - 1))
        size = 8
        value = 5 + self.current_wave
        self.enemies.append(Enemy(self.path_screen_coords, health, speed, size, value))

    def _find_closest_enemy(self, unit):
        closest_enemy = None
        min_dist = float('inf')
        unit_pos = self._iso_to_screen(*unit.grid_pos)
        for enemy in self.enemies:
            enemy_pos = self._iso_to_screen(*enemy.pos) if isinstance(enemy.pos[0], (int, float)) and len(enemy.pos) == 2 and not isinstance(enemy.pos[0], (list, tuple)) else enemy.pos
            dist = np.linalg.norm(np.array(unit_pos) - np.array(enemy_pos))
            if dist < unit.attack_range and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._iso_to_screen(c, r)
                points = [
                    (x, y),
                    (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
                    (x, y + self.TILE_HEIGHT_HALF * 2),
                    (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw path
        for r, c in self.path_grid_coords:
            if r < 0 or c < 0 or r >= self.GRID_WIDTH or c >= self.GRID_HEIGHT: continue
            self._render_iso_rect(self.screen, self.COLOR_PATH, (r, c), 0)

        # Draw base
        self._render_iso_rect(self.screen, self.COLOR_BASE_SHADOW, self.base_pos_grid, 15)
        self._render_iso_rect(self.screen, self.COLOR_BASE, self.base_pos_grid, 15, 5)
        self._render_health_bar(self._iso_to_screen(*self.base_pos_grid), self.base_health, self.max_base_health, 40)

        # Sort dynamic objects for correct rendering order
        render_queue = []
        for unit in self.units:
            render_queue.append(('unit', unit))
        for enemy in self.enemies:
            render_queue.append(('enemy', enemy))
        
        # Approximate y-coordinate for sorting
        def get_sort_key(item):
            obj = item[1]
            if isinstance(obj, Unit):
                return obj.grid_pos[0] + obj.grid_pos[1]
            else: # Enemy
                # Find closest path grid point to determine depth
                min_dist = float('inf')
                best_idx = 0
                for i, p_pos in enumerate(self.path_grid_coords):
                    dist = np.linalg.norm(np.array(self._iso_to_screen(*p_pos)) - np.array(obj.pos))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                return self.path_grid_coords[best_idx][0] + self.path_grid_coords[best_idx][1]

        render_queue.sort(key=get_sort_key)

        for type, obj in render_queue:
            if type == 'unit':
                self._render_iso_rect(self.screen, self.COLOR_UNIT_SHADOW, obj.grid_pos, 8)
                self._render_iso_rect(self.screen, self.COLOR_UNIT, obj.grid_pos, 8, 3)
            elif type == 'enemy':
                pos = obj.pos
                size = obj.size
                shadow_pos = (int(pos[0]), int(pos[1] + size))
                pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], int(size*1.2), int(size*0.6), self.COLOR_ENEMY_SHADOW)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(size), self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(size), self.COLOR_ENEMY)
                self._render_health_bar(pos, obj.health, obj.max_health, 20)
        
        # Draw cursor
        cursor_screen_pos = self._iso_to_screen(*self.cursor_pos)
        is_valid_tile = tuple(self.cursor_pos) not in self.path_grid_set and \
                        tuple(self.cursor_pos) != self.base_pos_grid and \
                        not any(u.grid_pos == self.cursor_pos for u in self.units)
        
        cursor_color = self.COLOR_CURSOR if is_valid_tile and self.resources >= self.unit_cost else self.COLOR_CURSOR_INVALID
        points = [
            (cursor_screen_pos[0], cursor_screen_pos[1]),
            (cursor_screen_pos[0] + self.TILE_WIDTH_HALF, cursor_screen_pos[1] + self.TILE_HEIGHT_HALF),
            (cursor_screen_pos[0], cursor_screen_pos[1] + self.TILE_HEIGHT_HALF * 2),
            (cursor_screen_pos[0] - self.TILE_WIDTH_HALF, cursor_screen_pos[1] + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.lines(self.screen, cursor_color, True, points, 2)

        # Draw projectiles and particles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, self.COLOR_PROJECTILE)
        for part in self.particles:
            pygame.draw.circle(self.screen, part.color, part.pos, max(0, int(part.size)))

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        texts = [
            f"WAVE: {self.current_wave}/10",
            f"HEALTH: {max(0, self.base_health)}",
            f"RESOURCES: {self.resources}",
            f"SCORE: {int(self.score)}"
        ]
        for i, text in enumerate(texts):
            rendered_text = self.font_small.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(rendered_text, (10 + i * 150, 10))
            
        if self.game_over:
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = self.COLOR_UNIT if self.win else self.COLOR_ENEMY
            rendered_text = self.font_large.render(msg, True, color)
            text_rect = rendered_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)
        elif self.wave_countdown > 0 and self.current_wave == 0:
            msg = f"Wave starting in: {self.wave_countdown // 30 + 1}"
            rendered_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = rendered_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)

    def _render_health_bar(self, pos, health, max_health, width):
        if health < max_health:
            bar_x = pos[0] - width // 2
            bar_y = pos[1] - 25
            health_ratio = max(0, health / max_health)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(width * health_ratio), 5))

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = int((grid_x - grid_y) * self.TILE_WIDTH_HALF + self.OFFSET_X)
        screen_y = int((grid_x + grid_y) * self.TILE_HEIGHT_HALF + self.OFFSET_Y)
        return screen_x, screen_y
    
    def _render_iso_rect(self, surface, color, grid_pos, height, y_offset=0):
        x, y = self._iso_to_screen(*grid_pos)
        y -= y_offset
        points = [
            (x, y - height),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - height),
            (x, y + self.TILE_HEIGHT_HALF * 2 - height),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - height)
        ]
        side1 = [
            (x, y), (x, y - height), 
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - height),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        side2 = [
            (x, y), (x, y - height),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF - height),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)
        
        pygame.gfxdraw.filled_polygon(surface, side1, darkest_color)
        pygame.gfxdraw.filled_polygon(surface, side2, darker_color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)
        
    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies) + (self.enemies_in_wave - self.enemies_spawned)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Use a different screen for display that can be resized
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH * 1.5, env.SCREEN_HEIGHT * 1.5), pygame.RESIZABLE)
    pygame.display.set_caption("Isometric Tower Defense")
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    while not done:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Actions
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Display the frame ---
        # The observation is (H, W, C), but pygame blit needs a surface
        # So we convert it back.
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Scale the surface to the display window size
        scaled_surface = pygame.transform.scale(frame_surface, display_screen.get_size())
        display_screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        
    env.close()