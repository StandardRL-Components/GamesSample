
# Generated: 2025-08-28T03:51:19.739834
# Source Brief: brief_02143.md
# Brief Index: 2143

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected tower. "
        "Hold Shift to cycle through tower types."
    )

    game_description = (
        "An isometric tower defense game. Defend your base from waves of enemies by "
        "strategically placing defensive towers. Earn resources by defeating enemies and "
        "survive for 20 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    # Game & World
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    WIN_WAVE = 20

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_PATH = (55, 60, 70)
    COLOR_BASE = (0, 255, 127)
    COLOR_BASE_DMG = (255, 70, 70)
    COLOR_ENEMY = (255, 59, 48)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_CURSOR_VALID = (0, 255, 127, 150)
    COLOR_CURSOR_INVALID = (255, 59, 48, 150)

    # Tower Specs
    TOWER_SPECS = {
        1: {"name": "Turret", "cost": 100, "range": 3.5, "damage": 10, "fire_rate": 0.8, "color": (0, 190, 255)},
        2: {"name": "Cannon", "cost": 250, "range": 2.5, "damage": 40, "fire_rate": 2.0, "color": (255, 149, 0)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        self.game_over_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.game_over_surface.fill((0, 0, 0, 180))

        self.np_random = None
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_result = 0 # -1 for loss, 1 for win

        self.base_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        self.spawn_pos = (1, self.GRID_HEIGHT // 2)
        self.max_base_health = 1000
        self.base_health = self.max_base_health
        self.last_base_hit_time = -1000

        self.resources = 250
        self.wave = 0
        self.wave_timer = 0
        self.time_between_waves = 5 * 30 # 5 seconds
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemy_spawn_timer = 0
        self.time_between_enemies = 1 * 30 # 1 second

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.grid[self.base_pos] = -1 # Mark base as unwalkable

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 1
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            reward += self._handle_input(movement, space_held, shift_held)
            self._update_wave_spawning()
            self._update_towers()
            reward += self._update_enemies()
            reward += self._update_projectiles()
            self._update_particles()

        self.steps += 1
        
        if not self.game_over:
            if self.base_health <= 0:
                self.game_over = True
                self.game_result = -1
                reward = -100
            elif self.wave > self.WIN_WAVE:
                self.game_over = True
                self.game_result = 1
                reward = 100
            elif self.steps >= self.MAX_STEPS:
                 self.game_over = True
                 self.game_result = -1
                 reward = -100

        terminated = self.game_over
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            self.selected_tower_type += 1
            if self.selected_tower_type > len(self.TOWER_SPECS):
                self.selected_tower_type = 1
            # sfx: cycle_weapon.wav

        # --- Place Tower ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self._is_placement_valid(self.cursor_pos, spec['cost']):
                self.resources -= spec['cost']
                reward -= spec['cost'] * 0.01
                self.grid[self.cursor_pos[0], self.cursor_pos[1]] = self.selected_tower_type
                
                tower_id = len(self.towers)
                self.towers.append({
                    "id": tower_id, "type": self.selected_tower_type, "spec": spec,
                    "pos": list(self.cursor_pos), "cooldown": 0
                })
                # sfx: place_tower.wav
                self._create_particles(self._iso_to_screen(*self.cursor_pos), 20, spec['color'])
                # Re-path all enemies
                for enemy in self.enemies:
                    enemy['path'] = self._find_path(tuple(np.round(enemy['grid_pos']).astype(int)), self.base_pos)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _update_wave_spawning(self):
        if len(self.enemies) == 0 and self.enemies_spawned >= self.enemies_in_wave:
            self.wave_timer += 1
            if self.wave_timer >= self.time_between_waves:
                if self.wave <= self.WIN_WAVE:
                    self._start_next_wave()
                    self.score += 5 # Wave survival bonus

    def _start_next_wave(self):
        self.wave += 1
        if self.wave > self.WIN_WAVE: return

        self.wave_timer = 0
        self.enemies_spawned = 0
        self.enemies_in_wave = 2 + self.wave * 2
        
        self.enemy_health_mult = (1.05) ** (self.wave - 1)
        self.enemy_speed_mult = (1.02) ** (self.wave - 1)

    def _spawn_enemy(self):
        self.enemies_spawned += 1
        path = self._find_path(self.spawn_pos, self.base_pos)
        if path:
            self.enemies.append({
                "pos": np.array(self._iso_to_screen(*self.spawn_pos), dtype=float),
                "grid_pos": np.array(self.spawn_pos, dtype=float),
                "health": 100 * self.enemy_health_mult,
                "max_health": 100 * self.enemy_health_mult,
                "speed": (0.03 + self.np_random.uniform(-0.005, 0.005)) * self.enemy_speed_mult,
                "damage": 10 * self.enemy_health_mult,
                "path": path,
                "path_index": 0
            })
            # sfx: enemy_spawn.wav

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1 / 30.0)
            if tower['cooldown'] <= 0:
                target = self._find_target(tower)
                if target:
                    tower['cooldown'] = tower['spec']['fire_rate']
                    self.projectiles.append({
                        "pos": np.array(self._iso_to_screen(*tower['pos']), dtype=float),
                        "target": target,
                        "speed": 8.0,
                        "damage": tower['spec']['damage'],
                        "color": tower['spec']['color']
                    })
                    # sfx: tower_fire.wav
    
    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if not enemy['path'] or enemy['path_index'] >= len(enemy['path']):
                # Reached base or got stuck
                dist_to_base = np.linalg.norm(np.array(enemy['grid_pos']) - np.array(self.base_pos))
                if dist_to_base < 1.5:
                    self.base_health -= enemy['damage']
                    self.last_base_hit_time = self.steps
                    self.enemies.remove(enemy)
                    # sfx: base_hit.wav
                    self._create_particles(self._iso_to_screen(*self.base_pos), 30, self.COLOR_BASE_DMG)
                continue

            target_grid_pos = enemy['path'][enemy['path_index']]
            target_screen_pos = np.array(self._iso_to_screen(*target_grid_pos))
            
            direction = target_screen_pos - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < enemy['speed'] * 30: # Close enough, advance to next path node
                enemy['path_index'] += 1
            else:
                enemy['pos'] += (direction / dist) * enemy['speed'] * 30 * (1/30.0) * 30
                
            enemy['grid_pos'][0] = (enemy['pos'][0] - self.ORIGIN_X) / self.TILE_WIDTH_HALF + (enemy['pos'][1] - self.ORIGIN_Y) / self.TILE_HEIGHT_HALF
            enemy['grid_pos'][1] = (enemy['pos'][1] - self.ORIGIN_Y) / self.TILE_HEIGHT_HALF - (enemy['pos'][0] - self.ORIGIN_X) / self.TILE_WIDTH_HALF
            enemy['grid_pos'] /= 2
        
        if self.enemies_spawned < self.enemies_in_wave:
            self.enemy_spawn_timer += 1
            if self.enemy_spawn_timer >= self.time_between_enemies:
                self.enemy_spawn_timer = 0
                self._spawn_enemy()
        return reward

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_pos = p['target']['pos']
            direction = target_pos - p['pos']
            dist = np.linalg.norm(direction)

            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                reward += 0.1 # Damage reward
                # sfx: enemy_hit.wav
                self._create_particles(p['pos'], 10, self.COLOR_ENEMY)
                if p['target']['health'] <= 0:
                    reward += 1.0 # Kill reward
                    self.resources += 25
                    self._create_particles(p['target']['pos'], 40, (255, 255, 0))
                    # sfx: enemy_explode.wav
                    self.enemies.remove(p['target'])
                self.projectiles.remove(p)
            else:
                p['pos'] += (direction / dist) * p['speed']
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self.screen.blit(self.game_over_surface, (0, 0))
            self._render_text(
                "GAME OVER", self.font_l, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20
            )
            win_text = "YOU WIN!" if self.game_result == 1 else "YOU LOSE"
            self._render_text(
                win_text, self.font_m, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20
            )

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_path()
        
        render_list = self.towers + self.enemies
        render_list.sort(key=lambda e: e.get('grid_pos', e.get('pos'))[1])

        self._render_base()
        self._render_cursor()

        for item in render_list:
            if 'spec' in item: # It's a tower
                self._render_tower(item)
            else: # It's an enemy
                self._render_enemy(item)

        for p in self.projectiles:
            self._render_projectile(p)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['life'] / 5)))

    def _render_grid_and_path(self):
        path = self._find_path(self.spawn_pos, self.base_pos)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._iso_to_screen(x, y)
                points = [
                    self._iso_to_screen(x, y),
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x, y + 1)
                ]
                color = self.COLOR_GRID
                if path and (x, y) in path:
                    color = self.COLOR_PATH
                pygame.gfxdraw.aapolygon(self.screen, points, color)
    
    def _render_base(self):
        x, y = self.base_pos
        base_color = self.COLOR_BASE
        if self.steps - self.last_base_hit_time < 15: # 0.5s flash
            base_color = self.COLOR_BASE_DMG
        
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, base_color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))

    def _render_cursor(self):
        x, y = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_valid = self._is_placement_valid(self.cursor_pos, spec['cost'])
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw range indicator
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        radius_x = spec['range'] * self.TILE_WIDTH_HALF
        radius_y = spec['range'] * self.TILE_HEIGHT_HALF
        pygame.gfxdraw.aaellipse(self.screen, int(center_x), int(center_y), int(radius_x), int(radius_y), (255,255,255,50))


    def _render_tower(self, tower):
        x, y = tower['pos']
        color = tower['spec']['color']
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        
        top_y = center_y - self.TILE_HEIGHT_HALF
        
        # Base
        base_points = [
            (center_x - self.TILE_WIDTH_HALF, center_y),
            (center_x, center_y + self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, center_y),
            (center_x, center_y - self.TILE_HEIGHT_HALF)
        ]
        darker_color = tuple(c*0.6 for c in color)
        pygame.gfxdraw.filled_polygon(self.screen, base_points, darker_color)
        
        # Top
        top_points = [
            (center_x - self.TILE_WIDTH_HALF, top_y),
            (center_x, top_y + self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, top_y),
            (center_x, top_y - self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, (255,255,255))
        
    def _render_enemy(self, enemy):
        pos = enemy['pos']
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 7, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 7, (0,0,0))
        
        # Health bar
        bar_w = 20
        bar_h = 4
        bar_x = pos[0] - bar_w / 2
        bar_y = pos[1] - 15
        health_pct = enemy['health'] / enemy['max_health']
        pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_projectile(self, p):
        x, y = int(p['pos'][0]), int(p['pos'][1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, 3, p['color'])
        pygame.gfxdraw.aacircle(self.screen, x, y, 3, (255, 255, 255))

    def _render_ui(self):
        # Base Health
        self._render_text(f"Base HP: {int(self.base_health)} / {self.max_base_health}", self.font_s, 10, 10, 'topleft')
        # Resources
        self._render_text(f"Resources: {self.resources}", self.font_s, 10, 30, 'topleft')
        # Wave
        self._render_text(f"Wave: {self.wave}", self.font_s, self.SCREEN_WIDTH - 10, 10, 'topright')
        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_s, self.SCREEN_WIDTH - 10, 30, 'topright')
        
        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self._render_text(f"Build: {spec['name']} (Cost: {spec['cost']})", self.font_m, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 15, 'midbottom')

    def _render_text(self, text, font, x, y, align="center"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "center": text_rect.center = (x, y)
        elif align == "topleft": text_rect.topleft = (x, y)
        elif align == "topright": text_rect.topright = (x, y)
        elif align == "midbottom": text_rect.midbottom = (x, y)
        
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "resources": self.resources}

    # --- Helper Functions ---
    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)
    
    def _find_path(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            (vx, vy), path = q.popleft()
            if (vx, vy) == end:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited and self.grid[nx, ny] == 0:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        return None # No path found

    def _is_placement_valid(self, pos, cost):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT): return False
        if self.grid[x, y] != 0: return False
        if self.resources < cost: return False
        if (x, y) == self.spawn_pos or (x,y) == self.base_pos: return False

        # Check if it blocks the path
        self.grid[x, y] = -2 # Tentative block
        path_exists = self._find_path(self.spawn_pos, self.base_pos) is not None
        self.grid[x, y] = 0 # Revert
        return path_exists

    def _find_target(self, tower):
        tower_pos = np.array(tower['pos'])
        in_range_enemies = []
        for enemy in self.enemies:
            dist = np.linalg.norm(tower_pos - enemy['grid_pos'])
            if dist <= tower['spec']['range']:
                in_range_enemies.append(enemy)
        
        if not in_range_enemies:
            return None
        
        # Target enemy closest to the base
        in_range_enemies.sort(key=lambda e: len(e['path']) - e['path_index'])
        return in_range_enemies[0]

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 25),
                "color": color
            })
    
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    while not done:
        movement = 0
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    print(f"Game Over! Final Score: {total_reward}, Wave: {info['wave']}")
    env.close()