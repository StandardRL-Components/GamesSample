
# Generated: 2025-08-27T18:44:24.298644
# Source Brief: brief_01924.md
# Brief Index: 1924

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press 'Shift' to cycle through tower types. "
        "Press 'Space' to build a tower at the cursor's location."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist isometric tower defense game. "
        "Place towers to defend your base from waves of enemies. "
        "Manage your funds and survive all the waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 15
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 5

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PATH = (50, 60, 80)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.TOWER_COLORS = [(80, 120, 255), (255, 180, 50)] # Blue, Yellow

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Isometric Projection Helpers ---
        self.tile_width_half = 16
        self.tile_height_half = 8
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80

        # --- Game Path ---
        self.path = self._create_path()

        # --- Entity Definitions ---
        self.TOWER_STATS = [
            {'cost': 25, 'range': 4, 'damage': 2, 'cooldown': 30, 'proj_speed': 5}, # Basic
            {'cost': 75, 'range': 6, 'damage': 5, 'cooldown': 50, 'proj_speed': 7}  # Advanced
        ]
        
        # --- Internal State ---
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.funds = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.available_tower_types = []
        self.wave_in_progress = False
        self.enemies_to_spawn = deque()
        self.spawn_timer = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.reward_this_step = 0

        # Initialize state variables
        self.reset()
        # self.validate_implementation() # Optional: for self-testing

    def _create_path(self):
        path = []
        for i in range(self.GRID_WIDTH):
            path.append((i, 3))
        for i in range(4, self.GRID_HEIGHT - 3):
            path.append((self.GRID_WIDTH - 1, i))
        for i in range(self.GRID_WIDTH - 1, -1, -1):
            path.append((i, self.GRID_HEIGHT - 4))
        return path

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.tile_width_half
        screen_y = self.origin_y + (grid_x + grid_y) * self.tile_height_half
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.funds = 80
        self.wave_number = 0
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.available_tower_types = [0]
        self.wave_in_progress = False
        self.enemies_to_spawn.clear()
        self.last_shift_press = False
        self.last_space_press = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        if self.wave_number > 2 and 1 not in self.available_tower_types:
            self.available_tower_types.append(1)

        self.wave_in_progress = True
        num_enemies = 5 + self.wave_number * 3
        enemy_health = 5 + (self.wave_number - 1) * 2
        enemy_speed = 0.5 + (self.wave_number - 1) * 0.1
        
        self.enemies_to_spawn.clear()
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({'health': enemy_health, 'speed': enemy_speed})
        self.spawn_timer = 0


    def step(self, action):
        movement, space_held_val, shift_held_val = action
        
        self.reward_this_step = -0.01 # Per-step penalty
        
        # --- 1. Handle Player Input ---
        self._handle_actions(movement, space_held_val, shift_held_val)

        # --- 2. Update Game Logic ---
        self._update_spawner()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- 3. Check Wave Completion ---
        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.reward_this_step += 1.0 # Wave completion bonus
            self.score += 100
            if self.wave_number < self.MAX_WAVES:
                self._start_next_wave()

        # --- 4. Check Termination Conditions ---
        terminated = False
        if self.base_health <= 0:
            self.reward_this_step -= 100 # Loss penalty
            self.score -= 1000
            terminated = True
        elif self.wave_number > self.MAX_WAVES and not self.enemies:
            self.reward_this_step += 100 # Win bonus
            self.score += 5000
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.steps += 1
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, movement, space_val, shift_val):
        space_pressed = space_val == 1
        shift_pressed = shift_val == 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Switch tower type (on press)
        if shift_pressed and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.available_tower_types)
        
        # Place tower (on press)
        if space_pressed and not self.last_space_press:
            selected_type_idx = self.available_tower_types[self.selected_tower_type]
            stats = self.TOWER_STATS[selected_type_idx]
            if self.funds >= stats['cost']:
                is_on_path = tuple(self.cursor_pos) in self.path
                is_occupied = any(t['pos'] == self.cursor_pos for t in self.towers)
                if not is_on_path and not is_occupied:
                    self.funds -= stats['cost']
                    self.towers.append({
                        'pos': list(self.cursor_pos),
                        'type': selected_type_idx,
                        'cooldown_left': 0,
                        'target': None
                    })
                    # sfx: place_tower.wav
        
        self.last_space_press = space_pressed
        self.last_shift_press = shift_pressed

    def _update_spawner(self):
        if not self.wave_in_progress or not self.enemies_to_spawn:
            return
        
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            spawn_data = self.enemies_to_spawn.popleft()
            start_pos = self.path[0]
            self.enemies.append({
                'pos': [float(start_pos[0]), float(start_pos[1])],
                'health': spawn_data['health'],
                'max_health': spawn_data['health'],
                'speed': spawn_data['speed'],
                'path_index': 1
            })
            self.spawn_timer = 40 # Time between spawns

    def _update_towers(self):
        for tower in self.towers:
            stats = self.TOWER_STATS[tower['type']]
            if tower['cooldown_left'] > 0:
                tower['cooldown_left'] -= 1
                continue

            # Find a target
            target = None
            min_dist_sq = (stats['range'] ** 2)
            for enemy in self.enemies:
                dist_sq = (tower['pos'][0] - enemy['pos'][0])**2 + (tower['pos'][1] - enemy['pos'][1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = enemy
            
            if target:
                tower['target'] = target
                tower['cooldown_left'] = stats['cooldown']
                start_pos = self._iso_to_screen(tower['pos'][0], tower['pos'][1])
                self.projectiles.append({
                    'start_pos': list(start_pos),
                    'pos': list(start_pos),
                    'target_enemy': target,
                    'damage': stats['damage'],
                    'speed': stats['proj_speed']
                })
                # sfx: shoot.wav

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target_enemy'] not in self.enemies:
                # Target is already dead
                self.projectiles.remove(proj)
                continue

            target_pos = self._iso_to_screen(proj['target_enemy']['pos'][0], proj['target_enemy']['pos'][1])
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                # Hit
                proj['target_enemy']['health'] -= proj['damage']
                if proj['target_enemy']['health'] <= 0:
                    # sfx: enemy_die.wav
                    self._create_explosion(proj['pos'], self.COLOR_ENEMY)
                    self.enemies.remove(proj['target_enemy'])
                    self.reward_this_step += 0.1
                    self.score += 10
                    self.funds += 5
                else:
                    # sfx: enemy_hit.wav
                    self._create_explosion(proj['pos'], self.COLOR_PROJECTILE, count=3, max_life=5)
                self.projectiles.remove(proj)
            else:
                # Move
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_index'] >= len(self.path):
                # Reached base
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                self._create_explosion(self._iso_to_screen(*self.path[-1]), self.COLOR_BASE_DMG, count=20)
                # sfx: base_damage.wav
                continue
            
            target_grid_pos = self.path[enemy['path_index']]
            dx = target_grid_pos[0] - enemy['pos'][0]
            dy = target_grid_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < 0.1:
                enemy['path_index'] += 1
            else:
                move_dist = min(dist, enemy['speed'] * 0.1) # Scale speed
                enemy['pos'][0] += (dx / dist) * move_dist
                enemy['pos'][1] += (dy / dist) * move_dist

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count=10, max_life=15, speed_range=(1, 3)):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.uniform(speed_range[0], speed_range[1])
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(max_life // 2, max_life),
                'max_life': max_life,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                points = [
                    self._iso_to_screen(x, y),
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x, y + 1)
                ]
                is_path = (x, y) in self.path
                pygame.draw.polygon(self.screen, self.COLOR_PATH if is_path else self.COLOR_GRID, points, 1)

        # Draw base
        base_pos = self.path[-1]
        self._render_iso_cube(self._iso_to_screen(base_pos[0], base_pos[1]), self.COLOR_BASE, 2)
        
        # Draw towers
        for tower in self.towers:
            pos = self._iso_to_screen(tower['pos'][0], tower['pos'][1])
            color = self.TOWER_COLORS[tower['type']]
            self._render_iso_cube(pos, color, 1)
            # Draw range indicator if cursor is on it
            if tower['pos'] == self.cursor_pos:
                range_px = self.TOWER_STATS[tower['type']]['range'] * self.tile_width_half * 1.5
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(range_px), (255,255,255,50))

        # Draw enemies
        for enemy in self.enemies:
            pos = self._iso_to_screen(enemy['pos'][0], enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1] - 5, 5, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1] - 5, 5, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 10
            pygame.draw.rect(self.screen, (50,50,50), (pos[0] - bar_w/2, pos[1] - 15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0] - bar_w/2, pos[1] - 15, int(bar_w * health_ratio), 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, 2, 2, 2, color)
            self.screen.blit(s, (pos[0]-2, pos[1]-2))

        # Draw cursor
        points = [
            self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1]),
            self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1]),
            self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1),
            self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1] + 1)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)


    def _render_iso_cube(self, pos, color, size=1):
        x, y = pos
        th, tw = self.tile_height_half, self.tile_width_half
        top_points = [(x, y), (x + tw, y + th), (x, y + 2*th), (x - tw, y + th)]
        
        darker_color = tuple(int(c*0.6) for c in color)
        darkest_color = tuple(int(c*0.4) for c in color)

        # Left face
        left_points = [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + size*th*2), (top_points[3][0], top_points[3][1] + size*th*2)]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, darker_color)
        # Right face
        right_points = [top_points[2], top_points[1], (top_points[1][0], top_points[1][1] + size*th*2), (top_points[2][0], top_points[2][1] + size*th*2)]
        pygame.gfxdraw.filled_polygon(self.screen, right_points, darkest_color)
        # Top face
        top_points_up = [(p[0], p[1] - size*th*2) for p in top_points]
        pygame.gfxdraw.filled_polygon(self.screen, top_points_up, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points_up, color)


    def _render_ui(self):
        # Base Health
        health_text = self.font_small.render(f"Base HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        pygame.draw.rect(self.screen, (50,50,50), (10, 30, 100, 10))
        health_ratio = self.base_health / 100
        health_color = self.COLOR_BASE if health_ratio > 0.3 else self.COLOR_BASE_DMG
        pygame.draw.rect(self.screen, health_color, (10, 30, int(100 * health_ratio), 10))

        # Funds
        funds_text = self.font_small.render(f"Funds: ${self.funds}", True, self.COLOR_TEXT)
        self.screen.blit(funds_text, (self.WIDTH - funds_text.get_width() - 10, 10))

        # Wave Info
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))
        
        # Selected Tower
        selected_type = self.available_tower_types[self.selected_tower_type]
        stats = self.TOWER_STATS[selected_type]
        tower_name = "Basic" if selected_type == 0 else "Advanced"
        tower_text = self.font_small.render(f"Build: {tower_name} (${stats['cost']})", True, self.TOWER_COLORS[selected_type])
        self.screen.blit(tower_text, (self.WIDTH - tower_text.get_width() - 10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "funds": self.funds,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    # To handle single key presses for step-by-step advance
    key_pressed_in_frame = False

    while not terminated:
        # For turn-based human play, we advance one step per key press
        action = [0, 0, 0] # Default to no-op
        key_pressed_in_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                key_pressed_in_frame = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                elif event.key == pygame.K_q: terminated = True
        
        # Only step the environment if a key was pressed
        if key_pressed_in_frame:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != -0.01:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Base HP: {info['base_health']}")

        # Render the current state regardless of stepping
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) 
        
        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(3000)

    env.close()