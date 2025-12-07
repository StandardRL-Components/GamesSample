
# Generated: 2025-08-28T05:08:12.549795
# Source Brief: brief_02521.md
# Brief Index: 2521

        
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


# --- Helper Classes for Game Entities ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, size_range, speed_range):
        self.x = x
        self.y = y
        self.vx = random.uniform(-speed_range, speed_range)
        self.vy = random.uniform(-speed_range, speed_range)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color_with_alpha = (*self.color, alpha)
            temp_surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, self.size, self.size))
            surface.blit(temp_surf, (self.x, self.y))

class Tower:
    """Represents a defensive tower."""
    TOWER_TYPES = [
        {"name": "Gatling", "cost": 75, "range": 80, "damage": 2, "cooldown": 10, "color": (0, 150, 255), "proj_speed": 5, "proj_size": 4},
        {"name": "Cannon", "cost": 150, "range": 120, "damage": 15, "cooldown": 45, "color": (255, 100, 0), "proj_speed": 4, "proj_size": 6},
    ]

    def __init__(self, grid_x, grid_y, tower_type_id, cell_size):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = (grid_x + 0.5) * cell_size
        self.y = (grid_y + 0.5) * cell_size
        self.type_id = tower_type_id
        self.stats = Tower.TOWER_TYPES[self.type_id]
        self.cooldown = 0
        self.target = None

    def update(self, enemies):
        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown == 0:
            self.find_target(enemies)
            if self.target:
                self.cooldown = self.stats["cooldown"]
                # // Sound: Tower fire
                return Projectile(self.x, self.y, self.target, self.stats)
        return None

    def find_target(self, enemies):
        self.target = None
        closest_dist = self.stats["range"] ** 2
        for enemy in enemies:
            dist_sq = (self.x - enemy.x)**2 + (self.y - enemy.y)**2
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                self.target = enemy

    def draw(self, surface, cell_size):
        color = self.stats["color"]
        pos = (int(self.x), int(self.y))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(cell_size * 0.4), color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(cell_size * 0.4), color)
        # Draw a small barrel indicating direction (if it has a target)
        if self.target and self.target.health > 0:
            angle = math.atan2(self.target.y - self.y, self.target.x - self.x)
            end_x = self.x + math.cos(angle) * cell_size * 0.5
            end_y = self.y + math.sin(angle) * cell_size * 0.5
            pygame.draw.line(surface, (255,255,255), pos, (int(end_x), int(end_y)), 2)


class Projectile:
    """Represents a projectile fired by a tower."""
    def __init__(self, x, y, target, tower_stats):
        self.x = x
        self.y = y
        self.target = target
        self.speed = tower_stats["proj_speed"]
        self.damage = tower_stats["damage"]
        self.color = (255, 255, 0)
        self.size = tower_stats["proj_size"]
        self.active = True

    def update(self):
        if not self.active or self.target.health <= 0:
            self.active = False
            return None
        
        angle = math.atan2(self.target.y - self.y, self.target.x - self.x)
        self.x += math.cos(angle) * self.speed
        self.y += math.sin(angle) * self.speed

        if math.hypot(self.x - self.target.x, self.y - self.target.y) < self.target.size / 2:
            self.active = False
            # // Sound: Projectile hit
            return self.target.take_damage(self.damage)
        return None

    def draw(self, surface):
        if self.active:
            pygame.draw.rect(surface, self.color, (int(self.x - self.size/2), int(self.y - self.size/2), self.size, self.size))


class Enemy:
    """Represents an enemy unit."""
    def __init__(self, path, cell_size, health, speed, wave_num):
        self.path = path
        self.cell_size = cell_size
        self.waypoint_index = 0
        self.x, self.y = self.get_pixel_coords(self.path[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = int(cell_size * 0.7)
        self.damage = 5 + wave_num # Damage to base
        self.color = self.get_wave_color(wave_num)

    def get_wave_color(self, wave_num):
        # Cycle through colors for visual distinction between waves
        colors = [(255, 0, 0), (255, 0, 128), (128, 0, 255), (0, 128, 255), (0, 255, 128), 
                  (128, 255, 0), (255, 255, 0), (255, 128, 0), (255, 64, 64), (255, 0, 0)]
        return colors[wave_num % len(colors)]

    def get_pixel_coords(self, grid_pos):
        return (grid_pos[0] + 0.5) * self.cell_size, (grid_pos[1] + 0.5) * self.cell_size

    def update(self):
        if self.waypoint_index >= len(self.path) - 1:
            return "reached_base"

        target_x, target_y = self.get_pixel_coords(self.path[self.waypoint_index + 1])
        angle = math.atan2(target_y - self.y, target_x - self.x)
        self.x += math.cos(angle) * self.speed
        self.y += math.sin(angle) * self.speed

        if math.hypot(self.x - target_x, self.y - target_y) < self.speed:
            self.waypoint_index += 1
        
        return None

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            # // Sound: Enemy destroyed
            return "killed"
        return "hit"

    def draw(self, surface):
        pos = (int(self.x), int(self.y))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.size // 2, self.color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.size // 2, self.color)
        # Health bar
        if self.health < self.max_health:
            health_pct = self.health / self.max_health
            bar_width = self.size
            bar_height = 4
            bar_x = self.x - bar_width / 2
            bar_y = self.y - self.size / 2 - 8
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0), (bar_x, bar_y, bar_width * health_pct, bar_height))


# --- Main Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to place tower, Shift to cycle tower type."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GAME_WIDTH = 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.GAME_WIDTH // self.GRID_SIZE
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 10
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_MONEY = 250
        self.WAVE_PREP_TIME = 90 # steps between waves

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PATH = (30, 30, 40)
        self.COLOR_BASE = (0, 255, 100)
        self.COLOR_UI_BG = (30, 30, 40)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        
        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game Path ---
        self.path = [(0, 2), (3, 2), (3, 5), (6, 5), (6, 2), (12, 2), (12, 8), (9, 8), (9, 11), (15, 11), (15, 16), (10, 16), (10, 19), (19, 19)]
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.base_health = 0
        self.money = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.is_wave_active = False
        self.game_over = False
        self.victory = False
        
        self.cursor_pos = [0,0]
        self.selected_tower_type = 0
        self.prev_shift_held = False
        
        self.grid = []
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.is_wave_active = False
        self.game_over = False
        self.victory = False

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tower_type = 0
        self.prev_shift_held = False

        self.grid = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        self._handle_actions(action)
        reward += self._update_game_state()
        
        term_reward, terminated = self._check_termination()
        reward += term_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        
        # Cycle tower type (on press, not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(Tower.TOWER_TYPES)
        self.prev_shift_held = shift_held

        # Place tower
        if space_held:
            x, y = self.cursor_pos
            cost = Tower.TOWER_TYPES[self.selected_tower_type]["cost"]
            is_on_path = any(self.cursor_pos == list(p) for p in self.path)

            if self.grid[y][x] == 0 and not is_on_path and self.money >= cost:
                self.money -= cost
                new_tower = Tower(x, y, self.selected_tower_type, self.CELL_SIZE)
                self.towers.append(new_tower)
                self.grid[y][x] = 1
                # // Sound: Tower placed
                for _ in range(20):
                    self.particles.append(Particle(new_tower.x, new_tower.y, new_tower.stats['color'], 20, (1, 3), 2))

    def _update_game_state(self):
        step_reward = 0.0

        # Wave management
        if not self.is_wave_active and not self.victory:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_wave()
        
        # Update towers and create projectiles
        new_projectiles = []
        for tower in self.towers:
            proj = tower.update(self.enemies)
            if proj:
                new_projectiles.append(proj)
        self.projectiles.extend(new_projectiles)

        # Update projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            result = proj.update()
            if result == "killed":
                step_reward += 0.1
                self.score += 10 * self.current_wave
                self.money += 5 + self.current_wave
                for _ in range(15):
                    self.particles.append(Particle(proj.target.x, proj.target.y, proj.target.color, 25, (2, 4), 3))
            elif result == "hit":
                for _ in range(5):
                    self.particles.append(Particle(proj.x, proj.y, (255, 200, 0), 10, (1, 2), 1))
            if not proj.active:
                projectiles_to_remove.append(proj)
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

        # Update enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                enemies_to_remove.append(enemy)
                continue
            result = enemy.update()
            if result == "reached_base":
                self.base_health = max(0, self.base_health - enemy.damage)
                enemies_to_remove.append(enemy)
                # // Sound: Base damage
                for _ in range(30):
                    self.particles.append(Particle(self.GAME_WIDTH - 10, enemy.y, (255, 50, 50), 40, (2, 5), 4))
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Check for wave completion
        if self.is_wave_active and not self.enemies:
            self.is_wave_active = False
            self.wave_timer = self.WAVE_PREP_TIME
            if self.current_wave < self.MAX_WAVES:
                step_reward += 1.0
                self.score += 100
                self.money += 50 + 10 * self.current_wave
            elif self.current_wave == self.MAX_WAVES:
                self.victory = True

        return step_reward

    def _spawn_wave(self):
        self.current_wave += 1
        self.is_wave_active = True
        num_enemies = 5 + self.current_wave * 2
        health = 10 * (1.1 ** (self.current_wave - 1))
        speed = 1.0 * (1.05 ** (self.current_wave - 1))
        
        for i in range(num_enemies):
            # Stagger spawn
            offset_path = [(p[0] - i*2, p[1]) for p in self.path]
            enemy = Enemy(offset_path, self.CELL_SIZE, health, speed, self.current_wave)
            self.enemies.append(enemy)

    def _check_termination(self):
        terminated = False
        reward = 0.0

        if self.base_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.victory:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        path_points = [( (p[0] + 0.5) * self.CELL_SIZE, (p[1] + 0.5) * self.CELL_SIZE) for p in self.path]
        if len(path_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_points, self.CELL_SIZE)

        # Draw base
        base_pos = self.path[-1]
        base_rect = pygame.Rect(base_pos[0]*self.CELL_SIZE, base_pos[1]*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen, self.CELL_SIZE)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
            
        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)
            
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw cursor and tower range
        cursor_color = self.COLOR_CURSOR
        tower_cost = Tower.TOWER_TYPES[self.selected_tower_type]['cost']
        is_on_path = any(self.cursor_pos == list(p) for p in self.path)
        can_afford = self.money >= tower_cost
        is_occupied = self.grid[self.cursor_pos[1]][self.cursor_pos[0]] == 1
        
        if is_occupied or not can_afford or is_on_path:
            cursor_color = (255, 0, 0)
            
        cx, cy = self.cursor_pos
        cursor_rect = (cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 2)

        # Draw tower range preview
        range_px = Tower.TOWER_TYPES[self.selected_tower_type]['range']
        center_px = ((cx + 0.5) * self.CELL_SIZE, (cy + 0.5) * self.CELL_SIZE)
        pygame.gfxdraw.aacircle(self.screen, int(center_px[0]), int(center_px[1]), range_px, (*cursor_color, 100))

    def _render_ui(self):
        ui_x = self.GAME_WIDTH
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, 0, self.WIDTH - ui_x, self.HEIGHT))
        
        y_pos = 20
        def draw_text(text, font, color, x, y, center=False):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            if center:
                rect.center = (x, y)
            else:
                rect.topleft = (x, y)
            self.screen.blit(surface, rect)
            return y + rect.height
        
        y_pos = draw_text(f"SCORE: {self.score}", self.font_medium, self.COLOR_UI_TEXT, ui_x + 20, y_pos) + 10
        y_pos = draw_text(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", self.font_medium, self.COLOR_UI_TEXT, ui_x + 20, y_pos)
        
        if not self.is_wave_active and not self.victory:
            next_wave_in = math.ceil(self.wave_timer)
            draw_text(f"Next wave in: {next_wave_in}", self.font_small, (255, 200, 0), ui_x + 20, y_pos + 10)
        
        y_pos += 40
        
        # Base health bar
        draw_text("BASE HEALTH", self.font_medium, self.COLOR_UI_TEXT, ui_x + 20, y_pos)
        y_pos += 25
        health_pct = self.base_health / self.INITIAL_BASE_HEALTH if self.INITIAL_BASE_HEALTH > 0 else 0
        bar_width = self.WIDTH - ui_x - 40
        pygame.draw.rect(self.screen, (50, 50, 50), (ui_x + 20, y_pos, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (ui_x + 20, y_pos, bar_width * health_pct, 20))
        y_pos += 30

        # Money
        y_pos = draw_text(f"MONEY: ${self.money}", self.font_medium, (255, 223, 0), ui_x + 20, y_pos) + 20
        
        # Selected Tower Info
        tower_info = Tower.TOWER_TYPES[self.selected_tower_type]
        draw_text("SELECTED TOWER", self.font_medium, self.COLOR_UI_TEXT, ui_x + 20, y_pos)
        y_pos += 25
        
        tower_color = tower_info['color'] if self.money >= tower_info['cost'] else (100, 100, 100)
        y_pos = draw_text(f"> {tower_info['name']}", self.font_large, tower_color, ui_x + 30, y_pos)
        y_pos += 10
        draw_text(f"Cost: ${tower_info['cost']}", self.font_small, self.COLOR_UI_TEXT, ui_x + 30, y_pos)
        y_pos += 18
        draw_text(f"Damage: {tower_info['damage']}", self.font_small, self.COLOR_UI_TEXT, ui_x + 30, y_pos)
        y_pos += 18
        draw_text(f"Range: {tower_info['range'] // self.CELL_SIZE}", self.font_small, self.COLOR_UI_TEXT, ui_x + 30, y_pos)
        y_pos += 18
        draw_text(f"Rate: {60 / tower_info['cooldown']:.1f}/s", self.font_small, self.COLOR_UI_TEXT, ui_x + 30, y_pos)

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.GAME_WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (0, 255, 0) if self.victory else (255, 0, 0)
            draw_text(msg, self.font_large, color, self.GAME_WIDTH // 2, self.HEIGHT // 2, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "money": self.money,
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")