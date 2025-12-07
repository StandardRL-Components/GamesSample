
# Generated: 2025-08-28T02:04:45.403640
# Source Brief: brief_01579.md
# Brief Index: 1579

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→↓↑ to select a build location. Space to build a Gatling Tower, Shift to build a Cannon Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic tower defense game. Place towers on the grid to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 50
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.FPS = 30
        self.MAX_STEPS = 4500 # 150 seconds
        self.MAX_WAVES = 5

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PATH = (30, 30, 45)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_TOWER_1 = (50, 100, 255)
        self.COLOR_TOWER_2 = (255, 200, 50)
        self.COLOR_PROJ_1 = (150, 200, 255)
        self.COLOR_PROJ_2 = (255, 230, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)

        # Game asset definitions
        self.TOWER_SPECS = {
            1: {"cost": 25, "range": 80, "fire_rate": 20, "damage": 10, "proj_speed": 8, "color": self.COLOR_TOWER_1, "proj_color": self.COLOR_PROJ_1},
            2: {"cost": 75, "range": 120, "fire_rate": 90, "damage": 50, "proj_speed": 10, "color": self.COLOR_TOWER_2, "proj_color": self.COLOR_PROJ_2},
        }

        # Define enemy path (grid coordinates)
        self.path_coords = [
            (-1, 1), (1, 1), (1, 6), (3, 6), (3, 1), (5, 1), (5, 6), (7, 6), (9, 6)
        ]
        self.path_pixels = [self._grid_to_pixel(r, c) for r, c in self.path_coords]
        self.base_pos = (7, 6) # Grid position of the base

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.money = 0
        self.base_health = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = None
        self.buildable_tiles = []
        self.selector_index = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 0
        self.enemies_to_spawn_in_wave = 0
        self.enemy_spawn_timer = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        # Initialize state variables
        self.reset()
    
    def _grid_to_pixel(self, r, c):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _get_buildable_tiles(self):
        path_set = set()
        for i in range(len(self.path_coords) - 1):
            r1, c1 = self.path_coords[i]
            r2, c2 = self.path_coords[i+1]
            for r in range(min(r1, r2), max(r1, r2) + 1):
                path_set.add((r, c1))
            for c in range(min(c1, c2), max(c1, c2) + 1):
                path_set.add((r2, c))
        
        buildable = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) not in path_set and (r,c) != self.base_pos:
                    buildable.append((r, c))
        return buildable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = self.np_random

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.money = 100
        self.base_health = 20
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.buildable_tiles = self._get_buildable_tiles()
        self.selector_index = 0
        self.wave_in_progress = False
        self.inter_wave_timer = self.FPS * 3
        self.enemies_to_spawn_in_wave = 0
        self.enemy_spawn_timer = 0
        self.game_over = False
        self.game_won = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.01 # Time penalty
        self.game_over = self._check_termination()

        if not self.game_over:
            reward += self._handle_input(movement, space_held, shift_held)
            
            # Update game logic
            wave_completion_reward = self._update_waves()
            reward += wave_completion_reward
            self._update_towers()
            reward += self._update_enemies()
            reward += self._update_projectiles()
            self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            if self.game_won:
                reward += 100
            elif self.base_health <= 0:
                reward -= 10 # Base health penalty is applied per hit
        
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        if len(self.buildable_tiles) > 0:
            if movement in [1, 4]: # Up, Right
                self.selector_index = (self.selector_index + 1) % len(self.buildable_tiles)
            elif movement in [2, 3]: # Down, Left
                self.selector_index = (self.selector_index - 1 + len(self.buildable_tiles)) % len(self.buildable_tiles)
        
        if (space_held or shift_held) and len(self.buildable_tiles) > 0:
            r, c = self.buildable_tiles[self.selector_index]
            
            if self.grid[r][c] == 0:
                tower_type = 1 if space_held else 2
                spec = self.TOWER_SPECS[tower_type]
                
                if self.money >= spec["cost"]:
                    self.money -= spec["cost"]
                    # SFX: build_tower.wav
                    self.towers.append({
                        "r": r, "c": c, "type": tower_type, "cooldown": 0,
                        "pos": self._grid_to_pixel(r, c)
                    })
                    self.grid[r][c] = tower_type
                    for _ in range(20):
                        self._create_particle(self._grid_to_pixel(r, c), spec["color"], 2, 20)
                    return 0 # No immediate reward for building
        return 0

    def _update_waves(self):
        reward = 0
        if self.wave_in_progress:
            if self.enemies_to_spawn_in_wave > 0 and self.enemy_spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn_in_wave -= 1
                self.enemy_spawn_timer = self.FPS // 2
            self.enemy_spawn_timer -= 1
            
            if self.enemies_to_spawn_in_wave == 0 and not self.enemies:
                self.wave_in_progress = False
                self.inter_wave_timer = self.FPS * 5
                self.score += self.wave_number * 10
                reward += 50 # Wave survived reward
        else:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0 and self.wave_number < self.MAX_WAVES:
                self._start_next_wave()
        return reward

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        self.enemies_to_spawn_in_wave = 10 + (self.wave_number - 1) * 5
        self.enemy_spawn_timer = 0
    
    def _spawn_enemy(self):
        health = 30 + (self.wave_number - 1) * 15
        speed = 1.0 + (self.wave_number - 1) * 0.1
        self.enemies.append({
            "pos": list(self.path_pixels[0]), "health": health, "max_health": health,
            "speed": speed, "path_index": 0, "value": 5 + self.wave_number
        })

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            spec = self.TOWER_SPECS[tower["type"]]
            target = None
            min_dist_sq = spec["range"] ** 2
            
            for enemy in self.enemies:
                dist_sq = (tower["pos"][0] - enemy["pos"][0])**2 + (tower["pos"][1] - enemy["pos"][1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = enemy
            
            if target:
                # SFX: fire_gatling.wav or fire_cannon.wav
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "pos": list(tower["pos"]), "target": target, "spec": spec
                })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] < len(self.path_pixels) - 1:
                target_pos = self.path_pixels[enemy["path_index"] + 1]
                dx, dy = target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy["speed"]:
                    enemy["pos"] = list(target_pos)
                    enemy["path_index"] += 1
                else:
                    enemy["pos"][0] += (dx / dist) * enemy["speed"]
                    enemy["pos"][1] += (dy / dist) * enemy["speed"]
            else:
                self.enemies.remove(enemy)
                self.base_health -= 1
                reward -= 10
                # SFX: base_damage.wav
                for _ in range(30):
                    self._create_particle(self._grid_to_pixel(*self.base_pos), self.COLOR_ENEMY, 3, 30)
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            spec = proj["spec"]
            target = proj["target"]
            
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            dx, dy = target["pos"][0] - proj["pos"][0], target["pos"][1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < spec["proj_speed"]:
                self.projectiles.remove(proj)
                target["health"] -= spec["damage"]
                reward += 0.1 # Hit reward
                # SFX: enemy_hit.wav
                self._create_particle(target["pos"], spec["proj_color"], 1, 15, count=10)

                if target["health"] <= 0:
                    self.enemies.remove(target)
                    reward += 1.0 # Kill reward
                    self.score += target["value"]
                    self.money += target["value"]
                    # SFX: enemy_explode.wav
                    self._create_particle(target["pos"], self.COLOR_ENEMY, 3, 25, count=30)
            else:
                proj["pos"][0] += (dx / dist) * spec["proj_speed"]
                proj["pos"][1] += (dy / dist) * spec["proj_speed"]
        return reward

    def _create_particle(self, pos, color, speed_max, life_max, count=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(life_max // 2, life_max),
                "max_life": life_max, "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_won = False
            return True
        if self.wave_number >= self.MAX_WAVES and not self.enemies and not self.wave_in_progress:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(len(self.path_pixels) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_pixels[i], self.path_pixels[i+1], self.CELL_SIZE)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = (self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        base_r, base_c = self.base_pos
        base_rect = (self.GRID_OFFSET_X + base_c * self.CELL_SIZE, self.GRID_OFFSET_Y + base_r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        if len(self.buildable_tiles) > 0 and not self.game_over:
            r, c = self.buildable_tiles[self.selector_index]
            rect = (self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            alpha = 128 + 127 * math.sin(self.steps * 0.2)
            sel_color = self.COLOR_SELECTOR
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((sel_color[0], sel_color[1], sel_color[2], 60))
            self.screen.blit(s, rect[:2])
            pygame.draw.rect(self.screen, (sel_color[0], sel_color[1], sel_color[2], alpha), rect, 3)

        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            px, py = tower["pos"]
            size = self.CELL_SIZE // 2
            pygame.draw.rect(self.screen, spec["color"], (px - size//2, py - size//2, size, size))
            pygame.draw.rect(self.screen, tuple(min(255, c*1.5) for c in spec["color"]), (px - size//2, py - size//2, size, size), 2)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), spec["range"], (*spec["color"], 50))

        for proj in self.projectiles:
            px, py = proj["pos"]
            pygame.draw.rect(self.screen, proj["spec"]["proj_color"], (px - 2, py - 2, 5, 5))

        for enemy in self.enemies:
            px, py = int(enemy["pos"][0]), int(enemy["pos"][1])
            radius = 10
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, tuple(min(255, c*1.2) for c in self.COLOR_ENEMY))
            health_ratio = max(0, enemy["health"] / enemy["max_health"])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (px - radius, py - radius - 8, radius * 2, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (px - radius, py - radius - 8, radius * 2 * health_ratio, 5))

        for p in self.particles:
            alpha = 255 * (p["life"] / p["max_life"])
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(5 * (p["life"] / p["max_life"]))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos[0]-size, pos[1]-size))

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        ui_panel.fill((20, 20, 30, 200))
        self.screen.blit(ui_panel, (0, 0))
        
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        money_text = self.font_small.render(f"$: {self.money}", True, self.COLOR_TEXT)
        self.screen.blit(money_text, (160, 10))

        wave_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}" if self.wave_in_progress else f"NEXT: {max(0, self.inter_wave_timer // self.FPS)}s"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (280, 10))

        health_text = self.font_small.render(f"BASE HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - 150, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    keys_held = {"up": False, "down": False, "left": False, "right": False, "space": False, "shift": False}
    last_move_time = 0
    MOVE_DELAY = 5 # frames

    live_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")

    while running:
        current_time = env.steps
        
        # Action processing with delay for movement
        movement = 0
        if current_time > last_move_time + MOVE_DELAY:
            active_keys = pygame.key.get_pressed()
            moved = False
            if active_keys[pygame.K_UP]: movement = 1; moved = True
            elif active_keys[pygame.K_DOWN]: movement = 2; moved = True
            elif active_keys[pygame.K_LEFT]: movement = 3; moved = True
            elif active_keys[pygame.K_RIGHT]: movement = 4; moved = True
            if moved:
                last_move_time = current_time

        space = 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0
        shift = 1 if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        live_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()