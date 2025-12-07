import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build the selected tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers. "
        "Survive 10 waves to win. If an enemy reaches your base, you lose."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        # FIX: pygame.display.set_mode must be called to initialize the display, even in dummy mode.
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.game_font_small = pygame.font.Font(None, 24)
        self.game_font_large = pygame.font.Font(None, 48)

        # --- Colors & Style ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_PATH = (100, 80, 30)
        self.COLOR_BASE = (30, 80, 100)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR_VALID = (50, 220, 50, 150)
        self.COLOR_CURSOR_INVALID = (220, 50, 50, 150)
        
        self.TOWER_SPECS = {
            1: {"name": "Gun", "cost": 100, "range": 80, "damage": 10, "fire_rate": 0.5, "color": (0, 255, 128)},
            2: {"name": "Cannon", "cost": 250, "range": 120, "damage": 50, "fire_rate": 1.5, "color": (255, 128, 0)},
            3: {"name": "Sniper", "cost": 300, "range": 250, "damage": 100, "fire_rate": 3.0, "color": (0, 128, 255)},
            4: {"name": "Slow", "cost": 150, "range": 60, "damage": 2, "fire_rate": 1.0, "color": (128, 0, 255), "slow_factor": 0.5, "slow_duration": 2.0}
        }

        # --- Game World ---
        self.grid_size = (16, 10)
        self.tile_width = 40
        self.tile_height = 20
        self.origin = pygame.Vector2(self.screen_width / 2 - self.tile_width / 2, 80)
        self._define_path_and_grid()

        # --- Game Constants ---
        self.MAX_STEPS = 15000 # Approx 8 minutes at 30fps
        self.TARGET_WAVES = 10
        self.INTERMISSION_TIME = 5.0 # seconds

        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.resources = 0
        self.wave_number = 0
        self.wave_state = "" # INTERMISSION, SPAWNING, ACTIVE
        self.wave_timer = 0.0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0.0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = (0, 0)
        self.selected_tower_type = 1
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        # self.validate_implementation() # Removed for submission

    def _define_path_and_grid(self):
        self.path = [
            (-1, 4), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2),
            (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 6), (8, 6),
            (9, 6), (10, 6), (11, 6), (11, 5), (11, 4), (11, 3), (12, 3),
            (13, 3), (14, 3), (15, 3), (16, 3)
        ]
        self.valid_placements = []
        path_set = set(self.path)
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                if (c, r) not in path_set:
                    self.valid_placements.append((c, r))

    def _iso_to_screen(self, c, r):
        x = self.origin.x + (c - r) * self.tile_width / 2
        y = self.origin.y + (c + r) * self.tile_height / 2
        return pygame.Vector2(x, y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.resources = 250
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.selected_tower_type = 1
        self.last_space_held = False
        self.last_shift_held = False

        self._start_intermission()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        dt = self.clock.tick(30) / 1000.0
        reward = -0.001 # Small time penalty

        self._handle_action(action)
        
        self._update_wave_system(dt)
        
        for tower in self.towers:
            new_projectiles = tower.update(dt, self.enemies)
            self.projectiles.extend(new_projectiles)

        for proj in self.projectiles[:]:
            if proj.update(dt):
                self.projectiles.remove(proj)
                reward += self._handle_hit(proj)
        
        for enemy in self.enemies[:]:
            if enemy.update(dt): # Reached end of path
                self.game_over = True
                self.win = False
            elif enemy.health <= 0:
                self.enemies.remove(enemy)
                self.resources += enemy.reward
                self.score += enemy.reward
                reward += 0.1
                self._create_particles(enemy.pos, (255,255,100), 20, 200, 0.5)

        for particle in self.particles[:]:
            if particle.update(dt):
                self.particles.remove(particle)

        reward += self._check_wave_completion()
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        self.steps += 1
        
        obs = self._get_observation()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        return obs, reward, terminated, truncated, self._get_info()

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos = (
                max(0, min(self.grid_size[0] - 1, self.cursor_pos[0] + dx)),
                max(0, min(self.grid_size[1] - 1, self.cursor_pos[1] + dy))
            )

        # Cycle tower type on shift press
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type % len(self.TOWER_SPECS)) + 1
        
        # Place tower on space press
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources >= spec["cost"]:
            is_valid_spot = self.cursor_pos in self.valid_placements
            is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
            
            if is_valid_spot and not is_occupied:
                self.resources -= spec["cost"]
                new_tower = Tower(self.cursor_pos, self.selected_tower_type, self.TOWER_SPECS, self)
                self.towers.append(new_tower)

    def _update_wave_system(self, dt):
        if self.wave_state == "INTERMISSION":
            self.wave_timer -= dt
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        elif self.wave_state == "SPAWNING":
            self.spawn_timer -= dt
            if self.spawn_timer <= 0 and self.enemies_to_spawn > 0:
                self.spawn_timer = 0.5 # Spawn every 0.5 seconds
                self.enemies_to_spawn -= 1
                
                health = 50 * (1.05 ** (self.wave_number - 1))
                speed = 40 * (1.02 ** (self.wave_number - 1))
                new_enemy = Enemy(self.path, health, speed, 10, self)
                self.enemies.append(new_enemy)

                if self.enemies_to_spawn == 0:
                    self.wave_state = "ACTIVE"

    def _handle_hit(self, projectile):
        if projectile.target and projectile.target.health > 0:
            projectile.target.take_damage(projectile.damage)
            self._create_particles(projectile.pos, projectile.color, 5, 100, 0.2)
            
            if "slow_factor" in projectile.spec:
                projectile.target.apply_slow(projectile.spec["slow_factor"], projectile.spec["slow_duration"])
            return 0.01
        return 0

    def _check_wave_completion(self):
        if self.wave_state == "ACTIVE" and not self.enemies:
            self.score += 100 * self.wave_number
            if self.wave_number < self.TARGET_WAVES:
                self._start_intermission()
                return 1.0 # Wave complete reward
            else: # Final wave beaten
                self.game_over = True
                self.win = True
        return 0.0

    def _start_intermission(self):
        self.wave_state = "INTERMISSION"
        self.wave_timer = self.INTERMISSION_TIME
        self.wave_number += 1
        if self.wave_number > self.TARGET_WAVES:
            return
        
        self.enemies_to_spawn = 4 + self.wave_number
        self.spawn_timer = 0.0

    def _start_next_wave(self):
        self.wave_state = "SPAWNING"
        self.spawn_timer = 0.0 # Start spawning immediately

    def _check_termination(self):
        if self.game_over:
            if self.win:
                return True, 10.0 # Win reward
            else:
                return True, -10.0 # Loss penalty
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, -5.0 # Timeout penalty
        return False, 0.0

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
            "wave": self.wave_number,
            "resources": self.resources,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn,
        }

    def _render_game(self):
        # Draw grid
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                pos = self._iso_to_screen(c, r)
                tile_points = [
                    (pos.x, pos.y + self.tile_height / 2),
                    (pos.x + self.tile_width / 2, pos.y),
                    (pos.x + self.tile_width, pos.y + self.tile_height / 2),
                    (pos.x + self.tile_width / 2, pos.y + self.tile_height)
                ]
                pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_GRID)

        # Draw path and base
        for i, (c, r) in enumerate(self.path):
            pos = self._iso_to_screen(c, r)
            tile_points = [
                (pos.x, pos.y + self.tile_height / 2),
                (pos.x + self.tile_width / 2, pos.y),
                (pos.x + self.tile_width, pos.y + self.tile_height / 2),
                (pos.x + self.tile_width / 2, pos.y + self.tile_height)
            ]
            color = self.COLOR_BASE if i == len(self.path) - 1 else self.COLOR_PATH
            pygame.gfxdraw.filled_polygon(self.screen, tile_points, color)
            pygame.gfxdraw.aapolygon(self.screen, tile_points, tuple(min(255, x + 20) for x in color[:3]))
        
        drawable_entities = self.towers + self.enemies
        drawable_entities.sort(key=lambda e: e.pos.y)
        for entity in drawable_entities:
            entity.draw(self.screen)

        for proj in self.projectiles:
            proj.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
        
        self._render_cursor()

    def _render_cursor(self):
        is_valid_spot = self.cursor_pos in self.valid_placements
        is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
        can_afford = self.resources >= self.TOWER_SPECS[self.selected_tower_type]["cost"]
        
        color = self.COLOR_CURSOR_VALID if is_valid_spot and not is_occupied and can_afford else self.COLOR_CURSOR_INVALID
        
        pos = self._iso_to_screen(*self.cursor_pos)
        tile_points = [
            (pos.x, pos.y + self.tile_height / 2),
            (pos.x + self.tile_width / 2, pos.y),
            (pos.x + self.tile_width, pos.y + self.tile_height / 2),
            (pos.x + self.tile_width / 2, pos.y + self.tile_height)
        ]
        
        # FIX: Create a proper transparent overlay surface.
        # This surface is transparent by default.
        cursor_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(cursor_surface, tile_points, color)
        self.screen.blit(cursor_surface, (0,0))

    def _render_ui(self):
        ui_bar = pygame.Surface((self.screen_width, 40))
        ui_bar.fill((15, 20, 25))
        
        score_text = self.game_font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        resources_text = self.game_font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        wave_text = self.game_font_small.render(f"Wave: {self.wave_number}/{self.TARGET_WAVES}", True, self.COLOR_TEXT)
        
        ui_bar.blit(score_text, (10, 10))
        ui_bar.blit(resources_text, (200, 10))
        ui_bar.blit(wave_text, (400, 10))
        
        self.screen.blit(ui_bar, (0, 0))

        bottom_bar = pygame.Surface((self.screen_width, 60))
        bottom_bar.fill((15, 20, 25))
        
        for i, spec in self.TOWER_SPECS.items():
            is_selected = i == self.selected_tower_type
            can_afford = self.resources >= spec['cost']
            
            box_rect = pygame.Rect(10 + (i - 1) * 150, 5, 140, 50)
            border_color = (255,255,255) if is_selected else (80,80,80)
            pygame.draw.rect(bottom_bar, (40,50,60), box_rect)
            pygame.draw.rect(bottom_bar, border_color, box_rect, 2)
            
            name_color = self.COLOR_TEXT if can_afford else (150,150,150)
            name_surf = self.game_font_small.render(f"{i}: {spec['name']}", True, name_color)
            cost_surf = self.game_font_small.render(f"Cost: {spec['cost']}", True, name_color)
            
            bottom_bar.blit(name_surf, (box_rect.x + 5, box_rect.y + 5))
            bottom_bar.blit(cost_surf, (box_rect.x + 5, box_rect.y + 25))
            pygame.draw.circle(bottom_bar, spec['color'], (box_rect.x + 120, box_rect.y + 25), 10)

        self.screen.blit(bottom_bar, (0, self.screen_height - 60))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.game_font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))
        
        elif self.wave_state == "INTERMISSION":
            msg = f"Wave {self.wave_number} starting in {int(self.wave_timer)+1}..."
            text_surf = self.game_font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

class Tower:
    def __init__(self, grid_pos, tower_type, specs, env):
        self.grid_pos = grid_pos
        self.type = tower_type
        self.spec = specs[tower_type]
        self.env = env
        
        screen_center = self.env._iso_to_screen(*grid_pos)
        self.pos = pygame.Vector2(screen_center.x + self.env.tile_width / 2, screen_center.y + self.env.tile_height / 2)
        
        self.fire_cooldown = 0.0
        self.target = None

    def update(self, dt, enemies):
        self.fire_cooldown = max(0, self.fire_cooldown - dt)
        
        if not self.target or self.target.health <= 0 or self.pos.distance_to(self.target.pos) > self.spec["range"]:
            self.target = self._find_target(enemies)
            
        if self.target and self.fire_cooldown == 0:
            self.fire_cooldown = self.spec["fire_rate"]
            return [Projectile(self.pos, self.target, self.spec, self.env)]
        return []

    def _find_target(self, enemies):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in enemies:
            dist = self.pos.distance_to(enemy.pos)
            if dist <= self.spec["range"] and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def draw(self, surface):
        base_pos = (int(self.pos.x), int(self.pos.y))
        pygame.gfxdraw.filled_circle(surface, base_pos[0], base_pos[1], 12, (80,80,80))
        pygame.gfxdraw.aacircle(surface, base_pos[0], base_pos[1], 12, (120,120,120))
        
        top_pos = (int(self.pos.x), int(self.pos.y - 8))
        pygame.gfxdraw.filled_circle(surface, top_pos[0], top_pos[1], 8, self.spec["color"])
        pygame.gfxdraw.aacircle(surface, top_pos[0], top_pos[1], 8, tuple(min(255, c+50) for c in self.spec["color"]))

class Enemy:
    def __init__(self, path, health, speed, reward, env):
        self.path = path
        self.path_index = 0
        self.health = health
        self.max_health = health
        self.speed_base = speed
        self.speed_modifier = 1.0
        self.reward = reward
        self.env = env
        
        self.pos = self.env._iso_to_screen(*self.path[0])
        self.pos.x += self.env.tile_width / 2
        self.pos.y += self.env.tile_height / 2
        
        self.slow_timer = 0.0
        self._set_target()

    def _set_target(self):
        if self.path_index + 1 < len(self.path):
            self.path_index += 1
            target_grid_pos = self.path[self.path_index]
            self.target_pos = self.env._iso_to_screen(*target_grid_pos)
            self.target_pos.x += self.env.tile_width / 2
            self.target_pos.y += self.env.tile_height / 2
        else:
            self.target_pos = None

    def update(self, dt):
        if self.slow_timer > 0:
            self.slow_timer -= dt
            if self.slow_timer <= 0:
                self.speed_modifier = 1.0

        if not self.target_pos:
            return True

        direction = self.target_pos - self.pos
        distance = direction.length()
        
        if distance < 2:
            self._set_target()
            if not self.target_pos:
                return True
            return self.update(dt)
        
        direction.normalize_ip()
        self.pos += direction * self.speed_base * self.speed_modifier * dt
        return False

    def take_damage(self, amount):
        self.health -= amount

    def apply_slow(self, factor, duration):
        self.speed_modifier = min(self.speed_modifier, factor)
        self.slow_timer = max(self.slow_timer, duration)

    def draw(self, surface):
        pos_int = (int(self.pos.x), int(self.pos.y))
        
        iso_height = 12
        iso_width = 8
        top_poly = [
            (pos_int[0], pos_int[1] - iso_height),
            (pos_int[0] + iso_width, pos_int[1] - iso_height - iso_width/2),
            (pos_int[0], pos_int[1] - iso_height - iso_width),
            (pos_int[0] - iso_width, pos_int[1] - iso_height - iso_width/2)
        ]
        color = self.env.COLOR_ENEMY if self.speed_modifier == 1.0 else (100, 100, 255)
        pygame.gfxdraw.filled_polygon(surface, top_poly, color)
        pygame.gfxdraw.aapolygon(surface, top_poly, tuple(min(255, c+30) for c in color))

        bar_width = 20
        bar_height = 4
        bar_pos_x = self.pos.x - bar_width / 2
        bar_pos_y = self.pos.y - iso_height - iso_width - 8
        
        health_ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, (50, 0, 0), (bar_pos_x, bar_pos_y, bar_width, bar_height))
        pygame.draw.rect(surface, (255, 0, 0), (bar_pos_x, bar_pos_y, bar_width * health_ratio, bar_height))

class Projectile:
    def __init__(self, start_pos, target, spec, env):
        self.pos = start_pos.copy()
        self.target = target
        self.speed = 300
        self.damage = spec["damage"]
        self.color = spec["color"]
        self.spec = spec
        self.env = env

    def update(self, dt):
        if not self.target or self.target.health <= 0:
            return True
        
        direction = self.target.pos - self.pos
        distance = direction.length()
        
        if distance < 5:
            return True
            
        direction.normalize_ip()
        self.pos += direction * self.speed * dt
        
        if not (0 <= self.pos.x < self.env.screen_width and 0 <= self.pos.y < self.env.screen_height):
            return True
            
        return False

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 3, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 3, (255,255,255))

class Particle:
    def __init__(self, pos, color, max_speed, lifetime, size):
        self.pos = pos.copy()
        self.vel = pygame.Vector2(random.uniform(-max_speed, max_speed), random.uniform(-max_speed, max_speed))
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.size = size

    def update(self, dt):
        self.pos += self.vel * dt
        self.vel *= 0.95
        self.lifetime -= dt
        return self.lifetime <= 0

    def draw(self, surface):
        life_ratio = self.lifetime / self.initial_lifetime
        current_size = int(self.size * life_ratio)
        if current_size > 0:
            pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), current_size)