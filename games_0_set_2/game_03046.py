
# Generated: 2025-08-27T22:12:30.706882
# Source Brief: brief_03046.md
# Brief Index: 3046

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A tower defense game where the player places towers to defend a central base
    from waves of enemies. The goal is to survive all 20 waves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    FPS = 30
    MAX_STEPS = 30 * 120 # 2 minutes max game time

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 50)
    COLOR_PATH = (60, 60, 70)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (255, 0, 0)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_HEALTH = (50, 200, 50)
    COLOR_CURSOR = (0, 200, 200)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # --- Tower Definitions ---
    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 100, "range": 100, "damage": 10, "fire_rate": 0.8, "color": (255, 200, 0), "projectile_speed": 8, "projectile_color": (255, 255, 150)},
        1: {"name": "Missile", "cost": 150, "range": 150, "damage": 25, "fire_rate": 2.0, "color": (200, 50, 255), "projectile_speed": 6, "projectile_color": (220, 150, 255)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.base_pos_grid = None
        self.base_pos_px = None
        self.path_waypoints = None
        self.grid = None
        self.cursor_pos = None
        self.selected_tower_type = None
        self.towers = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.base_health = None
        self.resources = None
        self.wave_number = None
        self.wave_timer = None
        self.enemies_in_wave = None
        self.spawn_timer = None
        self.wave_cleared_bonus = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.score = None
        self.prev_space_held = False
        self.prev_shift_held = False

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.base_pos_grid = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.base_pos_px = (
            self.base_pos_grid[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.base_pos_grid[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        )

        self._generate_path()
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        for x, y in self.path_waypoints:
            if 0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS:
                self.grid[x, y] = 1 # Path
        self.grid[self.base_pos_grid] = 2 # Base

        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.base_health = 100
        self.resources = 250
        
        self.wave_number = 0
        self.wave_timer = 3 * self.FPS  # 3 second countdown to first wave
        self.enemies_in_wave = 0
        self.spawn_timer = 0
        self.wave_cleared_bonus = False

        self.game_over = False
        self.win = False
        self.steps = 0
        self.score = 0
        
        self.prev_space_held = True # Prevent placing on first frame
        self.prev_shift_held = True # Prevent cycling on first frame

        return self._get_observation(), self._get_info()

    def _generate_path(self):
        """Generates a winding path for enemies."""
        path = []
        start_edge = self.np_random.integers(4)
        if start_edge == 0: # Top
            pos = [self.np_random.integers(1, self.GRID_COLS - 1), 0]
        elif start_edge == 1: # Bottom
            pos = [self.np_random.integers(1, self.GRID_COLS - 1), self.GRID_ROWS - 1]
        elif start_edge == 2: # Left
            pos = [0, self.np_random.integers(1, self.GRID_ROWS - 1)]
        else: # Right
            pos = [self.GRID_COLS - 1, self.np_random.integers(1, self.GRID_ROWS - 1)]
        
        path.append(tuple(pos))
        
        # Simple random walk towards the base
        while tuple(pos) != self.base_pos_grid:
            # Move towards base
            dx = self.base_pos_grid[0] - pos[0]
            dy = self.base_pos_grid[1] - pos[1]
            
            # Prioritize moving along the axis with greater distance
            if abs(dx) > abs(dy):
                pos[0] += np.sign(dx)
            elif abs(dy) > abs(dx):
                pos[1] += np.sign(dy)
            else: # Diagonal, pick one randomly
                if self.np_random.random() > 0.5:
                    pos[0] += np.sign(dx)
                else:
                    pos[1] += np.sign(dy)
            
            # Ensure we don't step out of bounds
            pos[0] = np.clip(pos[0], 0, self.GRID_COLS - 1)
            pos[1] = np.clip(pos[1], 0, self.GRID_ROWS - 1)
            
            if tuple(pos) not in path:
                path.append(tuple(pos))

        self.path_waypoints = list(reversed(path))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.001 # Small penalty per step to encourage speed

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        if movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        if movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        if movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # On key press (state change from not held to held)
        if space_held and not self.prev_space_held:
            reward += self._place_tower()
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_CYCLE

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic ---
        self._update_wave_spawner()
        reward += self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
            else:
                reward += -100
                self.score += -100
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self):
        x, y = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        if self.grid[x, y] == 0 and self.resources >= spec["cost"]: # Can only build on empty ground
            self.resources -= spec["cost"]
            self.grid[x, y] = 3 # Tower
            
            tower_pos_px = (x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2)
            
            new_tower = {
                "pos": tower_pos_px,
                "type": self.selected_tower_type,
                "cooldown": 0,
                **spec
            }
            self.towers.append(new_tower)
            # sfx: TOWER_PLACE
            self._create_particles(tower_pos_px, 20, spec["color"])
            return 0.5 # Small reward for placing a tower
        # sfx: UI_ERROR
        return 0

    def _update_wave_spawner(self):
        if self.win: return

        if self.enemies_in_wave == 0 and not self.enemies:
            if self.wave_cleared_bonus:
                self.resources += 100 + self.wave_number * 10
                self.wave_cleared_bonus = False
            
            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                self.wave_number += 1
                if self.wave_number > 20:
                    self.win = True
                    return
                self.enemies_in_wave = 10 + self.wave_number * 2
                self.spawn_timer = 0
                self.wave_cleared_bonus = True
                # sfx: WAVE_START
        
        if self.enemies_in_wave > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_in_wave -= 1
                self.spawn_timer = max(5, self.FPS * 0.5 - self.wave_number * 0.02)

    def _spawn_enemy(self):
        start_pos = (self.path_waypoints[0][0] * self.CELL_SIZE + self.CELL_SIZE // 2, 
                     self.path_waypoints[0][1] * self.CELL_SIZE + self.CELL_SIZE // 2)
        
        health = 50 + self.wave_number * 5
        speed = 1.0 + self.wave_number * 0.05
        
        self.enemies.append({
            "pos": list(start_pos),
            "path_idx": 0,
            "health": health,
            "max_health": health,
            "speed": speed,
        })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            # Find target: enemy closest to the base
            target = None
            min_dist_to_base = float('inf')
            
            for enemy in self.enemies:
                dist_to_tower = math.hypot(enemy["pos"][0] - tower["pos"][0], enemy["pos"][1] - tower["pos"][1])
                if dist_to_tower <= tower["range"]:
                    dist_to_base = len(self.path_waypoints) - enemy["path_idx"]
                    if dist_to_base < min_dist_to_base:
                        min_dist_to_base = dist_to_base
                        target = enemy
            
            if target:
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "target": target,
                    "speed": tower["projectile_speed"],
                    "damage": tower["damage"],
                    "color": tower["projectile_color"]
                })
                tower["cooldown"] = tower["fire_rate"] * self.FPS
                # sfx: TOWER_FIRE
        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies: # Target already dead
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj["speed"]:
                proj["target"]["health"] -= proj["damage"]
                # sfx: PROJECTILE_HIT
                self._create_particles(proj["pos"], 10, proj["color"])
                self.projectiles.remove(proj)
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Move along path
            if enemy["path_idx"] < len(self.path_waypoints) - 1:
                target_waypoint = self.path_waypoints[enemy["path_idx"] + 1]
                target_pos = (target_waypoint[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                              target_waypoint[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
                
                dx = target_pos[0] - enemy["pos"][0]
                dy = target_pos[1] - enemy["pos"][1]
                dist = math.hypot(dx, dy)

                if dist < enemy["speed"]:
                    enemy["pos"] = list(target_pos)
                    enemy["path_idx"] += 1
                else:
                    enemy["pos"][0] += (dx / dist) * enemy["speed"]
                    enemy["pos"][1] += (dy / dist) * enemy["speed"]
            else: # Reached base
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                # sfx: BASE_DAMAGE
                self._create_particles(self.base_pos_px, 30, self.COLOR_BASE_DMG)
                self.enemies.remove(enemy)
                reward -= 5 # Heavy penalty for letting an enemy through
                continue
            
            if enemy["health"] <= 0:
                reward += 1.0 # Reward for killing an enemy
                self.resources += 15
                # sfx: ENEMY_DEATH
                self._create_particles(enemy["pos"], 20, self.COLOR_ENEMY)
                self.enemies.remove(enemy)
        return reward
        
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 3
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        return self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_grid_and_path(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[x, y] == 1: # Path
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_BG, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_base(self):
        rect = (self.base_pos_grid[0] * self.CELL_SIZE, self.base_pos_grid[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, rect)
        pygame.draw.rect(self.screen, (255,255,255), rect, 2)


    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            # Draw range circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(tower["range"]), (100, 100, 100, 100))
            
            if tower["type"] == 0: # Cannon
                pygame.draw.rect(self.screen, tower["color"], (pos[0] - 10, pos[1] - 10, 20, 20))
                pygame.draw.circle(self.screen, self.COLOR_BG, pos, 5)
            elif tower["type"] == 1: # Missile
                pygame.draw.circle(self.screen, tower["color"], pos, 12)
                pygame.draw.circle(self.screen, self.COLOR_BG, pos, 8)
                pygame.draw.circle(self.screen, tower["color"], pos, 4)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = 10
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - size, pos[1] - size, size*2, size*2))
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = 20
            pygame.draw.rect(self.screen, (100,0,0), (pos[0] - bar_width//2, pos[1] - 18, bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_width//2, pos[1] - 18, int(bar_width * health_ratio), 4))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.circle(self.screen, proj["color"], pos, 4)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            life_ratio = p["life"] / 20.0
            size = int(3 * life_ratio)
            if size > 0:
                color = tuple(int(c * life_ratio) for c in p["color"])
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw tower preview
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_afford = self.resources >= spec["cost"]
        is_valid_spot = self.grid[x, y] == 0
        
        # Draw range preview
        preview_color = (c // 4 for c in spec["color"])
        pygame.gfxdraw.filled_circle(self.screen, rect[0] + self.CELL_SIZE//2, rect[1] + self.CELL_SIZE//2, int(spec["range"]), (*preview_color, 50))
        
        # Draw cursor box
        cursor_color = self.COLOR_CURSOR
        if not is_valid_spot or not can_afford:
            cursor_color = (150, 0, 0)
        pygame.draw.rect(self.screen, cursor_color, rect, 2)
        
    def _render_ui(self):
        # Health
        self._render_text(f"❤ {self.base_health}", (10, 10), self.font_large)
        # Resources
        self._render_text(f"$ {self.resources}", (10, 40), self.font_large)
        # Wave
        self._render_text(f"Wave: {self.wave_number}/20", (self.SCREEN_WIDTH - 150, 10), self.font_large)
        
        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self._render_text(f"Tower: {spec['name']}", (10, self.SCREEN_HEIGHT - 30), self.font_small)
        self._render_text(f"Cost: {spec['cost']}", (160, self.SCREEN_HEIGHT - 30), self.font_small)

        # Wave Timer
        if self.wave_timer > 0 and self.wave_number < 20:
            secs = math.ceil(self.wave_timer / self.FPS)
            msg = f"Next wave in {secs}..."
            if self.wave_number == 0:
                msg = f"First wave in {secs}..."
            
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)
            
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.win else "GAME OVER"
        text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset before observation
        self.reset()
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset return
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")