
# Generated: 2025-08-28T07:09:54.158744
# Source Brief: brief_03163.md
# Brief Index: 3163

        
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
        "Controls: Arrow keys to move the cursor. Space to place the selected tower. Shift to cycle tower types."
    )

    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of creeps moving along a procedurally generated path."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_CREEP = (230, 25, 75)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_CURSOR = (255, 255, 25, 100)
        self.COLOR_CURSOR_INVALID = (230, 25, 75, 100)

        # Tower definitions
        self.TOWER_TYPES = [
            {'name': 'Cannon', 'cost': 100, 'range': 80, 'damage': 5, 'fire_rate': 1.0, 'color': (0, 130, 200)},
            {'name': 'Sniper', 'cost': 250, 'range': 160, 'damage': 20, 'fire_rate': 0.4, 'color': (255, 140, 0)},
        ]
        
        # Game constants
        self.MAX_STEPS = 2000
        self.INITIAL_GOLD = 300
        self.CREEP_SPEED = 1.0
        self.CREEP_RADIUS = 8
        self.PROJECTILE_SPEED = 8.0

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.creeps = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.path = []
        self.grid = []
        self.gold = 0
        self.wave = 0
        self.creeps_to_spawn = []
        self.spawn_timer = 0
        self.creeps_defeated = 0
        self.total_creeps_in_wave = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.creeps = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.gold = self.INITIAL_GOLD
        self.wave = 0
        
        self.cursor_pos = [self.GRID_W // 4, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_path()
        self.grid = [[True for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        for i in range(len(self.path) - 1):
            p1 = pygame.math.Vector2(self.path[i])
            p2 = pygame.math.Vector2(self.path[i+1])
            dist = int(p1.distance_to(p2))
            for j in range(dist):
                p = p1.lerp(p2, j / dist)
                gx, gy = int(p.x // self.GRID_SIZE), int(p.y // self.GRID_SIZE)
                if 0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H:
                    self.grid[gx][gy] = False # Path is not buildable

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # 1. Handle player input
        reward += self._handle_input(action)

        # 2. Update game logic
        self._spawn_creeps()
        self._update_towers()
        reward += self._update_projectiles()
        loss_condition, wave_cleared = self._update_creeps()
        self._update_particles()
        
        # 3. Check for termination
        terminated = False
        if loss_condition:
            reward = -100
            self.game_over = True
            terminated = True
        elif wave_cleared:
            reward += 100
            self._start_next_wave()
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_path(self):
        self.path = []
        start_y = self.np_random.integers(self.HEIGHT // 4, self.HEIGHT * 3 // 4)
        self.path.append((0, start_y))
        
        current_pos = pygame.math.Vector2(self.path[0])
        direction = pygame.math.Vector2(1, 0)
        
        while current_pos.x < self.WIDTH:
            segment_length = self.np_random.integers(80, 200)
            end_pos = current_pos + direction * segment_length
            
            if end_pos.x > self.WIDTH:
                end_pos.x = self.WIDTH
            
            end_pos.y = np.clip(end_pos.y, self.GRID_SIZE, self.HEIGHT - self.GRID_SIZE)
            self.path.append((int(end_pos.x), int(end_pos.y)))
            current_pos = end_pos
            
            turn = self.np_random.choice([-1, 1])
            direction = direction.rotate(turn * 90)

    def _start_next_wave(self):
        self.wave += 1
        self.creeps_defeated = 0
        num_creeps = 5 + self.wave * 2
        self.total_creeps_in_wave = num_creeps
        creep_health = 10 + self.wave * 5
        
        self.creeps_to_spawn = []
        for i in range(num_creeps):
            self.creeps_to_spawn.append({'health': creep_health, 'spawn_delay': i * 30})
        
        self.spawn_timer = 0

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
        
        # Place tower (on press)
        if space_held and not self.prev_space_held:
            tower_def = self.TOWER_TYPES[self.selected_tower_idx]
            gx, gy = self.cursor_pos
            if self.gold >= tower_def['cost'] and self.grid[gx][gy]:
                self.gold -= tower_def['cost']
                world_x = gx * self.GRID_SIZE + self.GRID_SIZE // 2
                world_y = gy * self.GRID_SIZE + self.GRID_SIZE // 2
                self.towers.append({
                    'pos': pygame.math.Vector2(world_x, world_y),
                    'type': self.selected_tower_idx,
                    'cooldown': 0,
                })
                self.grid[gx][gy] = False # Mark cell as occupied
                # SFX: Place Tower
                self._create_particles(pygame.math.Vector2(world_x, world_y), tower_def['color'], 20, 3)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return 0 # No immediate reward for input actions

    def _spawn_creeps(self):
        if not self.creeps_to_spawn:
            return
            
        self.spawn_timer += 1
        if self.spawn_timer >= self.creeps_to_spawn[0]['spawn_delay']:
            creep_info = self.creeps_to_spawn.pop(0)
            self.creeps.append({
                'pos': pygame.math.Vector2(self.path[0]),
                'health': creep_info['health'],
                'max_health': creep_info['health'],
                'path_idx': 1,
            })
            # SFX: Creep Spawn

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            tower_def = self.TOWER_TYPES[tower['type']]
            target = None
            
            # Target creep closest to the end of the path
            best_dist = -1
            for creep in self.creeps:
                dist_to_tower = tower['pos'].distance_to(creep['pos'])
                if dist_to_tower <= tower_def['range']:
                    # Heuristic for "closest to end": path_idx and distance to next waypoint
                    dist_along_path = creep['path_idx'] * 1000 - creep['pos'].distance_to(self.path[creep['path_idx']])
                    if dist_along_path > best_dist:
                        best_dist = dist_along_path
                        target = creep
            
            if target and tower['cooldown'] <= 0:
                self.projectiles.append({
                    'pos': tower['pos'].copy(),
                    'target': target,
                    'damage': tower_def['damage'],
                    'color': tower_def['color']
                })
                tower['cooldown'] = 60 / tower_def['fire_rate'] # 60 FPS assumption
                # SFX: Tower Fire

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.creeps: # Target already destroyed
                self.projectiles.remove(proj)
                continue
            
            direction = (proj['target']['pos'] - proj['pos']).normalize()
            proj['pos'] += direction * self.PROJECTILE_SPEED
            
            if proj['pos'].distance_to(proj['target']['pos']) < self.CREEP_RADIUS:
                proj['target']['health'] -= proj['damage']
                reward += 0.1 # Reward for hitting
                # SFX: Projectile Hit
                self._create_particles(proj['pos'], self.COLOR_CREEP, 5, 1)
                self.projectiles.remove(proj)
        return reward

    def _update_creeps(self):
        loss = False
        for creep in self.creeps[:]:
            if creep['health'] <= 0:
                self.gold += 10 + self.wave
                self.score += 1 # Reward for defeating
                self.creeps_defeated += 1
                self.creeps.remove(creep)
                # SFX: Creep Destroyed
                self._create_particles(creep['pos'], self.COLOR_CREEP, 30, 2)
                continue

            target_pos = pygame.math.Vector2(self.path[creep['path_idx']])
            if creep['pos'].distance_to(target_pos) < self.CREEP_SPEED * 1.5:
                creep['path_idx'] += 1
                if creep['path_idx'] >= len(self.path):
                    self.creeps.remove(creep)
                    loss = True # Reached the base
                    # SFX: Base Damaged
                    self._create_particles(self.path[-1], self.COLOR_BASE, 50, 5)
                    continue
                target_pos = pygame.math.Vector2(self.path[creep['path_idx']])
            
            direction = (target_pos - creep['pos']).normalize()
            creep['pos'] += direction * self.CREEP_SPEED

        wave_cleared = not self.creeps and not self.creeps_to_spawn
        return loss, wave_cleared

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_factor
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.integers(1, 4)
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
            "gold": self.gold,
            "wave": self.wave,
            "creeps_remaining": self.total_creeps_in_wave - self.creeps_defeated
        }

    def _render_game(self):
        # Path
        if len(self.path) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path, 1)
        
        # Base
        base_pos = self.path[-1]
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(base_pos[0]), int(base_pos[1])), 12)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), 12, self.COLOR_BASE)

        # Towers and range indicators
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower['type']]
            pos_int = (int(tower['pos'].x), int(tower['pos'].y))
            pygame.draw.circle(self.screen, tower_def['color'], pos_int, 10)
            pygame.draw.circle(self.screen, (255,255,255), pos_int, 4)

        # Creeps
        for creep in self.creeps:
            pos_int = (int(creep['pos'].x), int(creep['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_CREEP, pos_int, self.CREEP_RADIUS)
            # Health bar
            health_ratio = creep['health'] / creep['max_health']
            bar_w = self.CREEP_RADIUS * 2
            bar_h = 4
            bar_x = creep['pos'].x - self.CREEP_RADIUS
            bar_y = creep['pos'].y - self.CREEP_RADIUS - 8
            pygame.draw.rect(self.screen, (255,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,255,0), (bar_x, bar_y, bar_w * health_ratio, bar_h))

        # Projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj['color'], (int(proj['pos'].x), int(proj['pos'].y)), 4)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'].x), int(proj['pos'].y), 4, proj['color'])
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_ui(self):
        # Cursor and range indicator
        gx, gy = self.cursor_pos
        cursor_rect = pygame.Rect(gx * self.GRID_SIZE, gy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        tower_def = self.TOWER_TYPES[self.selected_tower_idx]
        is_valid_placement = self.grid[gx][gy] and self.gold >= tower_def['cost']
        cursor_color = self.COLOR_CURSOR if is_valid_placement else self.COLOR_CURSOR_INVALID

        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, cursor_rect.topleft)
        
        # Range indicator for cursor
        range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        center_x = gx * self.GRID_SIZE + self.GRID_SIZE // 2
        center_y = gy * self.GRID_SIZE + self.GRID_SIZE // 2
        pygame.draw.circle(range_surf, (*tower_def['color'], 50), (center_x, center_y), tower_def['range'], 0)
        pygame.draw.circle(range_surf, (*tower_def['color'], 150), (center_x, center_y), tower_def['range'], 1)
        self.screen.blit(range_surf, (0,0))

        # UI Text
        gold_text = self.font_m.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))
        
        wave_text = self.font_m.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        creeps_rem = self.total_creeps_in_wave - self.creeps_defeated
        creeps_text = self.font_m.render(f"REMAINING: {creeps_rem}", True, self.COLOR_TEXT)
        self.screen.blit(creeps_text, (self.WIDTH - creeps_text.get_width() - 10, self.HEIGHT - creeps_text.get_height() - 10))

        # Selected tower info
        tower_info_text = self.font_s.render(f"Selected: {tower_def['name']} (Cost: {tower_def['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (10, self.HEIGHT - tower_info_text.get_height() - 10))

        if self.game_over:
            outcome_text_str = "WAVE CLEARED" if self.creeps_defeated == self.total_creeps_in_wave and not self.steps >= self.MAX_STEPS else "GAME OVER"
            outcome_text = self.font_l.render(outcome_text_str, True, self.COLOR_TEXT)
            text_rect = outcome_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(outcome_text, text_rect)

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