
# Generated: 2025-08-28T05:19:43.479279
# Source Brief: brief_02569.md
# Brief Index: 2569

        
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
        "Controls: Arrow keys to move cursor. Space to place a block. Shift to cycle block type."
    )

    game_description = (
        "Defend your fortress from waves of enemies by strategically placing walls and turrets on a grid."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SQUARE_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SQUARE_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SQUARE_SIZE
        self.MAX_WAVES = 20
        self.MAX_STEPS = 1000
        self.FORTRESS_START_HEALTH = 50
        self.TURRET_RANGE = 5
        self.TURRET_COOLDOWN = 3 # steps

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_FORTRESS = (0, 200, 100)
        self.COLOR_FORTRESS_DMG = (255, 100, 0)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.BLOCK_TYPES = {
            1: {"name": "WALL", "color": (50, 100, 200)},
            2: {"name": "TURRET", "color": (150, 50, 200)},
        }

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # State variables (initialized in reset)
        self.grid = None
        self.enemies = None
        self.fortress_pos = None
        self.fortress_health = None
        self.wave = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.cursor_pos = None
        self.selected_block_type_idx = None
        self.unlocked_block_types = None
        self.wave_cleared = None
        self.particles = None
        self.turret_cooldowns = None
        self.fortress_damage_flash = 0

        self.np_random = None
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.fortress_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.grid[self.fortress_pos] = -1  # Fortress marker

        self.fortress_health = self.FORTRESS_START_HEALTH
        self.wave = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = np.array([0, 0])
        self.selected_block_type_idx = 0
        self.unlocked_block_types = [1] # Start with WALL
        self.wave_cleared = True
        self.particles = []
        self.turret_cooldowns = {} # (x, y) -> cooldown_timer

        self.enemies = []
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # 1. Handle player action
        reward += self._handle_input(action)
        
        # 2. Update game state
        reward += self._update_game_state()

        # 3. Check for wave clear
        if self.wave_cleared is False and not self.enemies:
            self.wave_cleared = True
            reward += 1.0
            # Unlock TURRET at wave 5
            if self.wave == 4 and 2 not in self.unlocked_block_types:
                self.unlocked_block_types.append(2)
            
            if self.wave < self.MAX_WAVES:
                self._spawn_wave()
            else: # Game won
                self.game_over = True

        # 4. Check for termination
        terminated = self.game_over or self.fortress_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Loss or timeout
            reward = -100.0
            self.game_over = True
        elif self.wave > self.MAX_WAVES and self.game_over: # Win
            reward = 100.0
        
        self.score += max(0, reward) # Score only tracks positive achievements

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, [0, 0], [self.GRID_W - 1, self.GRID_H - 1])

        # Cycle block type
        if shift_held:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.unlocked_block_types)
            # sound: block_cycle.wav

        # Place block
        if space_held:
            x, y = self.cursor_pos
            if self.grid[x, y] == 0:  # Can only place on empty tiles
                block_type_id = self.unlocked_block_types[self.selected_block_type_idx]
                self.grid[x, y] = block_type_id
                if block_type_id == 2: # Turret
                    self.turret_cooldowns[(x, y)] = 0
                reward -= 0.01
                self._create_particles((x + 0.5) * self.GRID_SQUARE_SIZE, (y + 0.5) * self.GRID_SQUARE_SIZE, 10, self.BLOCK_TYPES[block_type_id]["color"])
                # sound: place_block.wav
        return reward

    def _update_game_state(self):
        reward = 0
        
        # Update turret cooldowns
        for pos in self.turret_cooldowns:
            if self.turret_cooldowns[pos] > 0:
                self.turret_cooldowns[pos] -= 1
        
        # Turrets fire
        for r, c in np.argwhere(self.grid == 2): # Find all turrets
            if self.turret_cooldowns.get((r,c), 0) > 0:
                continue

            target = None
            min_dist = self.TURRET_RANGE ** 2
            for enemy in self.enemies:
                dist_sq = (enemy['pos'][0] - r)**2 + (enemy['pos'][1] - c)**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                target['health'] -= 1
                self.turret_cooldowns[(r, c)] = self.TURRET_COOLDOWN
                # sound: laser_shoot.wav
                self._create_particles(
                    (target['pos'][0] + 0.5) * self.GRID_SQUARE_SIZE,
                    (target['pos'][1] + 0.5) * self.GRID_SQUARE_SIZE,
                    15, self.COLOR_PROJECTILE, lifespan=10
                )

        # Process enemy health and deaths
        new_enemies = []
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                reward += 0.1
                self._create_particles(
                    (enemy['pos'][0] + 0.5) * self.GRID_SQUARE_SIZE,
                    (enemy['pos'][1] + 0.5) * self.GRID_SQUARE_SIZE,
                    30, self.COLOR_ENEMY
                )
                # sound: enemy_explode.wav
            else:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        # Enemies move
        for enemy in self.enemies:
            path = self._find_path(tuple(enemy['pos']), self.fortress_pos)
            if path and len(path) > 1:
                for _ in range(enemy['speed']):
                    if len(path) > 1:
                        next_pos = path.pop(1)
                        enemy['pos'] = np.array(next_pos)
                    else:
                        break

        # Enemies attack fortress
        surviving_enemies = []
        for enemy in self.enemies:
            if tuple(enemy['pos']) == self.fortress_pos:
                self.fortress_health -= 1
                self.fortress_damage_flash = 5 # frames
                # sound: fortress_hit.wav
            else:
                surviving_enemies.append(enemy)
        self.enemies = surviving_enemies
        
        return reward

    def _find_path(self, start, end):
        if start == end:
            return [start]
            
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            (vx, vy), path = q.popleft()
            
            # Order neighbors: N, S, W, E
            neighbors = [(vx, vy-1), (vx, vy+1), (vx-1, vy), (vx+1, vy)]
            
            for nx, ny in neighbors:
                if (nx, ny) == end:
                    return path + [(nx, ny)]
                
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[nx, ny] <= 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return None # No path found

    def _spawn_wave(self):
        self.wave += 1
        self.wave_cleared = False
        num_enemies = self.wave
        enemy_speed = 1 + (self.wave // 5)
        enemy_health = 1 + (self.wave // 10)

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0:   # Top
                pos = [self.np_random.integers(self.GRID_W), 0]
            elif side == 1: # Bottom
                pos = [self.np_random.integers(self.GRID_W), self.GRID_H - 1]
            elif side == 2: # Left
                pos = [0, self.np_random.integers(self.GRID_H)]
            else:           # Right
                pos = [self.GRID_W - 1, self.np_random.integers(self.GRID_H)]
            
            # Ensure spawn point is not the fortress
            if tuple(pos) == self.fortress_pos:
                pos[0] = (pos[0] + 1) % self.GRID_W

            self.enemies.append({
                'pos': np.array(pos, dtype=int),
                'health': enemy_health,
                'speed': enemy_speed,
            })

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
            "wave": self.wave,
            "health": self.fortress_health,
            "enemies": len(self.enemies)
        }

    def _render_game(self):
        # Grid lines
        for x in range(0, self.WIDTH, self.GRID_SQUARE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SQUARE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for r, c in np.argwhere(self.grid > 0):
            block_type_id = self.grid[r, c]
            color = self.BLOCK_TYPES[block_type_id]["color"]
            rect = (r * self.GRID_SQUARE_SIZE, c * self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE)
            pygame.draw.rect(self.screen, color, rect)

        # Fortress
        fx, fy = self.fortress_pos
        f_rect = (fx * self.GRID_SQUARE_SIZE, fy * self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, f_rect)
        if self.fortress_damage_flash > 0:
            s = pygame.Surface((self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE), pygame.SRCALPHA)
            s.fill((255, 0, 0, 128))
            self.screen.blit(s, (f_rect[0], f_rect[1]))
            self.fortress_damage_flash -= 1

        # Enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            e_rect = (ex * self.GRID_SQUARE_SIZE, ey * self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e_rect)
        
        # Particles
        self._update_and_draw_particles()

        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = (cx * self.GRID_SQUARE_SIZE, cy * self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE)
        s = pygame.Surface((self.GRID_SQUARE_SIZE, self.GRID_SQUARE_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (cursor_rect[0], cursor_rect[1]))

    def _render_ui(self):
        # Score
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_m.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Health Bar
        health_ratio = max(0, self.fortress_health / self.FORTRESS_START_HEALTH)
        health_bar_width = 150
        health_bar_rect = pygame.Rect(self.WIDTH // 2 - health_bar_width // 2, 10, health_bar_width * health_ratio, 20)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH // 2 - health_bar_width // 2, 10, health_bar_width, 20), 1)

        # Selected Block
        block_id = self.unlocked_block_types[self.selected_block_type_idx]
        block_info = self.BLOCK_TYPES[block_id]
        block_text = self.font_s.render(f"SELECT: {block_info['name']}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (10, self.HEIGHT - block_text.get_height() - 10))
        pygame.draw.rect(self.screen, block_info['color'], (10 + block_text.get_width() + 10, self.HEIGHT - 25, 15, 15))

        if self.game_over:
            outcome_text = "VICTORY!" if self.wave > self.MAX_WAVES else "GAME OVER"
            color = self.COLOR_FORTRESS if self.wave > self.MAX_WAVES else self.COLOR_ENEMY
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            go_text_surf = self.font_m.render(outcome_text, True, color)
            go_rect = go_text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(go_text_surf, go_rect)

    def _create_particles(self, x, y, count, color, lifespan=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [x, y], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha)
                )
        self.particles = active_particles

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Grid Defense")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop
    while not terminated:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        clock.tick(10) # Control the speed of manual play

    pygame.quit()