
# Generated: 2025-08-27T20:38:47.526481
# Source Brief: brief_02532.md
# Brief Index: 2532

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    A tower defense game where the player places blocks to defend a central core from waves of enemies.
    The player controls a cursor to select a grid location, cycles through block types, and places them.
    Enemies follow a path towards the core, and different blocks have different effects (blocking, slowing, attacking).
    The goal is to survive 20 waves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move cursor. Space to place block. Shift to cycle block type."
    )

    game_description = (
        "Defend your fortress from waves of enemies by strategically placing defensive blocks."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
    WIN_WAVE = 20
    WAVE_PREP_TIME = 30 * 5 # 5 seconds

    # Core
    CORE_HEALTH_MAX = 20

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 55)
    COLOR_CORE = (255, 50, 50)
    COLOR_CORE_GLOW = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    
    # Block types
    BLOCK_TYPES = {
        "WALL": {"color": (100, 120, 200), "cost": 10},
        "SLOW": {"color": (230, 210, 80), "cost": 20, "range": 2.5, "effect": 0.5},
        "TURRET": {"color": (200, 80, 220), "cost": 30, "range": 4.5, "cooldown": 30, "damage": 1},
    }
    BLOCK_TYPE_NAMES = list(BLOCK_TYPES.keys())

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 48)

        # State variables are initialized in reset()
        self.grid = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.turret_cooldowns = None
        self.core_pos_grid = None
        self.core_health = None
        self.resources = None
        self.wave_number = None
        self.wave_timer = None
        self.enemies_in_wave = None
        self.enemies_spawned = None
        self.spawn_timer = None
        self.cursor_pos = None
        self.selected_block_idx = None
        self.flow_field = None
        self.last_space_held = False
        self.last_shift_held = False
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int)
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.turret_cooldowns = {}
        
        self.core_pos_grid = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.core_health = self.CORE_HEALTH_MAX
        self.resources = 60
        
        self.wave_number = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.spawn_timer = 0
        
        self.cursor_pos = [self.GRID_COLS // 2 - 5, self.GRID_ROWS // 2]
        self.selected_block_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._calculate_flow_field()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        if movement == 2: self.cursor_pos[1] += 1
        if movement == 3: self.cursor_pos[0] -= 1
        if movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        
        # Cycle block type (on key press)
        if shift_held and not self.last_shift_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPE_NAMES)
        
        # Place block (on key press)
        if space_held and not self.last_space_held:
            x, y = self.cursor_pos
            block_name = self.BLOCK_TYPE_NAMES[self.selected_block_idx]
            cost = self.BLOCK_TYPES[block_name]['cost']
            is_valid_pos = self.grid[x, y] == -1 and (x, y) != self.core_pos_grid
            
            if self.resources >= cost and is_valid_pos:
                self.grid[x, y] = self.selected_block_idx
                self.resources -= cost
                reward -= 0.01 # Small penalty for placing
                if block_name == "TURRET":
                    self.turret_cooldowns[(x, y)] = 0
                self._calculate_flow_field()
                # sfx: place_block.wav
        
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        if not self.game_over:
            self._update_wave_logic()
            reward += self._update_turrets()
            self._update_projectiles()
            self._update_enemies()
        
        self._update_particles()
        
        # --- Calculate Reward & Termination ---
        # Rewards for destroying enemies and core damage are handled in their respective update functions
        
        terminated = self.game_over
        if not terminated:
            if self.win:
                reward += 100
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                reward -= 100 # Penalty for running out of time
                terminated = True
                self.game_over = True

        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Sub-functions ---

    def _update_wave_logic(self):
        # If wave is over, start prep timer for the next one
        if self.enemies_spawned >= self.enemies_in_wave and not self.enemies:
            if self.wave_number > 0 and self.wave_timer == self.WAVE_PREP_TIME:
                self.score += 1 # Wave survived reward
                self.resources += 20 + self.wave_number * 5 # Resource bonus
                if self.wave_number >= self.WIN_WAVE:
                    self.win = True
                    self.game_over = True
                    return

            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                self.wave_timer = self.WAVE_PREP_TIME
                self._start_new_wave()
        
        # Spawn enemies during a wave
        if self.enemies_spawned < self.enemies_in_wave:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.spawn_timer = max(5, 30 - self.wave_number) # Spawn faster in later waves

    def _start_new_wave(self):
        self.enemies_in_wave = 5 + self.wave_number * 2
        self.enemies_spawned = 0

    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: x, y = self.np_random.integers(self.GRID_COLS), 0
        elif side == 1: x, y = self.np_random.integers(self.GRID_COLS), self.GRID_ROWS - 1
        elif side == 2: x, y = 0, self.np_random.integers(self.GRID_ROWS)
        else: x, y = self.GRID_COLS - 1, self.np_random.integers(self.GRID_ROWS)
        
        pixel_pos = [x * self.CELL_SIZE + self.CELL_SIZE/2, y * self.CELL_SIZE + self.CELL_SIZE/2]
        
        health = 5 + self.wave_number // 2
        speed = 0.8 + (self.wave_number // 5) * 0.1
        
        self.enemies.append({
            "pos": np.array(pixel_pos, dtype=float),
            "health": health,
            "max_health": health,
            "speed": speed,
            "slow_timer": 0
        })
        self.enemies_spawned += 1

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            # Apply slow effect
            current_speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                current_speed *= self.BLOCK_TYPES["SLOW"]["effect"]

            # Movement based on flow field
            grid_x, grid_y = int(enemy["pos"][0] // self.CELL_SIZE), int(enemy["pos"][1] // self.CELL_SIZE)
            
            if 0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS:
                if (grid_x, grid_y) == self.core_pos_grid:
                    self.core_health -= 1
                    self.enemies.remove(enemy)
                    self._create_particles(enemy["pos"], self.COLOR_CORE, 20)
                    # sfx: core_damage.wav
                    if self.core_health <= 0:
                        self.score -= 100
                        self.game_over = True
                    continue

                best_dir = self._get_best_direction(grid_x, grid_y)
                enemy["pos"] += best_dir * current_speed

    def _update_turrets(self):
        reward = 0
        for (tx, ty), block_idx in np.ndenumerate(self.grid):
            if block_idx == self.BLOCK_TYPE_NAMES.index("TURRET"):
                self.turret_cooldowns[(tx, ty)] -= 1
                if self.turret_cooldowns[(tx, ty)] <= 0:
                    target = self._find_target((tx, ty), "TURRET")
                    if target:
                        self.turret_cooldowns[(tx, ty)] = self.BLOCK_TYPES["TURRET"]["cooldown"]
                        start_pos = np.array([tx * self.CELL_SIZE + self.CELL_SIZE/2, ty * self.CELL_SIZE + self.CELL_SIZE/2])
                        self.projectiles.append({"pos": start_pos, "target": target, "speed": 5})
                        # sfx: turret_fire.wav

            elif block_idx == self.BLOCK_TYPE_NAMES.index("SLOW"):
                targets = self._find_targets_in_range((tx, ty), "SLOW")
                for enemy in targets:
                    enemy["slow_timer"] = 2 # Apply slow for 2 frames
        return reward

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = proj["target"]["pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < proj["speed"]:
                proj["target"]["health"] -= self.BLOCK_TYPES["TURRET"]["damage"]
                self._create_particles(proj["pos"], (255, 255, 255), 5)
                # sfx: enemy_hit.wav
                if proj["target"]["health"] <= 0:
                    self.score += 0.1 # Kill reward
                    self._create_particles(proj["target"]["pos"], (100, 255, 100), 30)
                    # sfx: enemy_destroy.wav
                    self.enemies.remove(proj["target"])
                self.projectiles.remove(proj)
            else:
                proj["pos"] += (direction / dist) * proj["speed"]

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _calculate_flow_field(self):
        self.flow_field = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int)
        q = deque([self.core_pos_grid])
        self.flow_field[self.core_pos_grid] = 0
        
        while q:
            x, y = q.popleft()
            dist = self.flow_field[x, y]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    is_obstacle = self.grid[nx, ny] != -1
                    if not is_obstacle and self.flow_field[nx, ny] == -1:
                        self.flow_field[nx, ny] = dist + 1
                        q.append((nx, ny))

    def _get_best_direction(self, x, y):
        best_dir = np.array([0.0, 0.0])
        min_dist = self.flow_field[x, y]
        if min_dist == -1: return best_dir # Stranded

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                if self.flow_field[nx, ny] != -1 and self.flow_field[nx, ny] < min_dist:
                    min_dist = self.flow_field[nx, ny]
                    best_dir = np.array([dx, dy], dtype=float)
        
        # Normalize diagonal movement
        if np.linalg.norm(best_dir) > 1:
            best_dir /= np.linalg.norm(best_dir)
            
        return best_dir

    def _find_target(self, pos, block_type):
        min_dist = float('inf')
        best_target = None
        radius_sq = (self.BLOCK_TYPES[block_type]["range"] * self.CELL_SIZE) ** 2
        px, py = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE

        for enemy in self.enemies:
            dist_sq = (enemy["pos"][0] - px)**2 + (enemy["pos"][1] - py)**2
            if dist_sq < radius_sq and dist_sq < min_dist:
                min_dist = dist_sq
                best_target = enemy
        return best_target

    def _find_targets_in_range(self, pos, block_type):
        targets = []
        radius_sq = (self.BLOCK_TYPES[block_type]["range"] * self.CELL_SIZE) ** 2
        px, py = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE

        for enemy in self.enemies:
            dist_sq = (enemy["pos"][0] - px)**2 + (enemy["pos"][1] - py)**2
            if dist_sq < radius_sq:
                targets.append(enemy)
        return targets

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "color": color,
                "life": self.np_random.integers(10, 20)
            })

    # --- Rendering Sub-functions ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_blocks()
        self._render_core()
        self._render_projectiles()
        self._render_enemies()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _render_core(self):
        cx, cy = self.core_pos_grid
        px, py = cx * self.CELL_SIZE, cy * self.CELL_SIZE
        
        # Glow effect
        health_ratio = self.core_health / self.CORE_HEALTH_MAX
        glow_radius = int(self.CELL_SIZE * 0.8 * (0.5 + health_ratio * 0.5))
        glow_color = tuple(int(c * health_ratio) for c in self.COLOR_CORE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, glow_radius, glow_color)
        
        # Core square
        pygame.draw.rect(self.screen, self.COLOR_CORE, (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))

    def _render_blocks(self):
        for (x, y), block_idx in np.ndenumerate(self.grid):
            if block_idx != -1:
                block_name = self.BLOCK_TYPE_NAMES[block_idx]
                info = self.BLOCK_TYPES[block_name]
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                pygame.draw.rect(self.screen, info["color"], (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
                if 'range' in info:
                    radius = int(info['range'] * self.CELL_SIZE)
                    color = info['color'] + (30,) # Add alpha
                    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, color, (radius, radius), radius)
                    self.screen.blit(s, (px + self.CELL_SIZE//2 - radius, py + self.CELL_SIZE//2 - radius))


    def _render_enemies(self):
        for enemy in self.enemies:
            px, py = int(enemy["pos"][0]), int(enemy["pos"][1])
            color = (50, 200, 50) if enemy["slow_timer"] > 0 else (200, 100, 50)
            pygame.draw.circle(self.screen, color, (px, py), self.CELL_SIZE // 3)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_w = self.CELL_SIZE * 0.8
            bar_h = 3
            bar_x = px - bar_w / 2
            bar_y = py - self.CELL_SIZE * 0.6
            pygame.draw.rect(self.screen, (200, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            px, py = int(proj["pos"][0]), int(proj["pos"][1])
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"]
            pygame.gfxdraw.pixel(self.screen, int(p["pos"][0]), int(p["pos"][1]), color + (alpha,))

    def _render_cursor(self):
        x, y = self.cursor_pos
        px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (px, py, self.CELL_SIZE, self.CELL_SIZE), 2)

    def _render_ui(self):
        # Wave info
        wave_text = f"Wave: {self.wave_number}"
        if self.enemies_spawned >= self.enemies_in_wave and not self.enemies and not self.win:
             wave_text += f" (Next in {self.wave_timer//30 + 1}s)"
        ui_wave = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(ui_wave, (10, 10))

        # Resources
        ui_res = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(ui_res, (self.SCREEN_WIDTH - ui_res.get_width() - 10, 10))

        # Core Health
        ui_core = self.font_small.render(f"Core Health: {self.core_health}", True, self.COLOR_TEXT)
        self.screen.blit(ui_core, (self.SCREEN_WIDTH // 2 - ui_core.get_width() // 2, 10))
        
        # Selected Block
        block_name = self.BLOCK_TYPE_NAMES[self.selected_block_idx]
        block_info = self.BLOCK_TYPES[block_name]
        ui_block = self.font_small.render(f"Selected: {block_name} (Cost: {block_info['cost']})", True, self.COLOR_TEXT)
        pygame.draw.rect(self.screen, block_info['color'], (10, 30, 16, 16))
        self.screen.blit(ui_block, (30, 28))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_CURSOR if self.win else self.COLOR_CORE)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "core_health": self.core_health,
            "resources": self.resources,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Keyboard to Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()