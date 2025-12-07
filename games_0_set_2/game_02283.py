
# Generated: 2025-08-28T04:20:29.489095
# Source Brief: brief_02283.md
# Brief Index: 2283

        
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

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press space to place the selected tower. "
        "Hold shift to cycle through tower types."
    )

    game_description = (
        "A classic tower defense game. Strategically place towers to defend your base from waves of invading aliens. "
        "Survive 10 waves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 2500 # Increased from 1000 to allow for longer games
        self.MAX_WAVES = 10
        self.STARTING_BASE_HEALTH = 100

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_PATH = (45, 55, 75)
        self.COLOR_BASE = (0, 100, 200)
        self.COLOR_BASE_STROKE = (100, 180, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_HEALTH_BAR = (40, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (200, 40, 80)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)

        # --- Tower & Alien Definitions ---
        self.TOWER_SPECS = {
            0: {"name": "Cannon", "range": 100, "damage": 10, "fire_rate": 30, "color": (0, 150, 255), "proj_speed": 8},
            1: {"name": "Laser", "range": 80, "damage": 1.5, "fire_rate": 1, "color": (255, 0, 100), "proj_speed": 0},
        }
        self.ALIEN_BASE_HEALTH = 20
        self.ALIEN_BASE_SPEED = 1.0

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.current_wave = 0
        self.wave_spawning = False
        self.wave_spawn_timer = 0
        self.wave_aliens_to_spawn = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.step_reward = 0.0
        self.alien_path = []
        self.path_grid_coords = set()

        self._define_path()
        self.reset()
        
        # This is a dummy call to initialize the RNG if it's not already.
        # super().reset() in our reset method handles seeding properly.
        self.np_random = np.random.default_rng()


    def _define_path(self):
        self.path_grid_coords = set()
        path_points = [
            (0, 2), (2, 2), (2, 5), (5, 5), (5, 1), (9, 1), 
            (9, 7), (12, 7), (12, 3), (self.GRID_W-1, 3)
        ]
        
        for i in range(len(path_points) - 1):
            p1_x, p1_y = path_points[i]
            p2_x, p2_y = path_points[i+1]
            
            x, y = p1_x, p1_y
            self.path_grid_coords.add((x, y))
            while (x, y) != (p2_x, p2_y):
                if x < p2_x: x += 1
                elif x > p2_x: x -= 1
                if y < p2_y: y += 1
                elif y > p2_y: y -= 1
                self.path_grid_coords.add((x, y))

        self.alien_path = []
        for gx, gy in self.path_grid_coords:
             self.alien_path.append(self._grid_to_pixel(gx, gy))

        # A more precise path for aliens to follow
        self.alien_waypoints = [self._grid_to_pixel(p[0], p[1]) for p in path_points]
        self.base_pos_grid = (self.GRID_W // 2, self.GRID_H - 1)
        self.alien_waypoints.append(self._grid_to_pixel(self.base_pos_grid[0], self.base_pos_grid[1]))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = self.STARTING_BASE_HEALTH
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.current_wave = 0
        self.wave_spawning = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2 - 2]
        self.selected_tower_type = 0
        self.last_shift_press = True
        self.last_space_press = True
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0

        if self.game_over:
            return self._get_observation(), self.step_reward, True, False, self._get_info()
        
        self.steps += 1
        self._handle_input(action)
        self._update_game_state()

        terminated = self._check_termination()
        reward = self.step_reward

        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_press = shift_held

        # Place tower (on press)
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

    def _place_tower(self):
        gx, gy = self.cursor_pos
        is_on_path = (gx, gy) in self.path_grid_coords
        is_on_base = (gx, gy) == self.base_pos_grid
        is_occupied = any(t['grid_pos'] == [gx, gy] for t in self.towers)

        if not is_on_path and not is_occupied and not is_on_base:
            # Sound: tower_place.wav
            self.towers.append({
                "grid_pos": [gx, gy],
                "pixel_pos": self._grid_to_pixel(gx, gy),
                "type": self.selected_tower_type,
                "spec": self.TOWER_SPECS[self.selected_tower_type],
                "cooldown": 0,
                "target": None
            })

    def _update_game_state(self):
        self._update_wave_spawner()
        self._update_towers()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        if not self.wave_spawning and not self.aliens and not self.game_over:
            if self.current_wave >= self.MAX_WAVES:
                self.win = True
            else:
                self.step_reward += 1.0 # Wave clear bonus
                self.score += 50 * self.current_wave
                self._start_next_wave()
    
    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_aliens_to_spawn = 2 + self.current_wave
        self.wave_spawning = True
        self.wave_spawn_timer = 60 # Ticks before first spawn

    def _update_wave_spawner(self):
        if not self.wave_spawning:
            return
        
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.wave_aliens_to_spawn > 0:
            self.wave_spawn_timer = max(10, 45 - self.current_wave * 2) # Spawn faster on later waves
            self.wave_aliens_to_spawn -= 1
            
            health = self.ALIEN_BASE_HEALTH * (1 + (self.current_wave - 1) * 0.1)
            speed = self.ALIEN_BASE_SPEED * (1 + (self.current_wave - 1) * 0.05)
            
            self.aliens.append({
                "pos": list(self.alien_waypoints[0]),
                "health": health,
                "max_health": health,
                "speed": speed,
                "waypoint_idx": 1,
                "id": self.np_random.random()
            })
            # Sound: alien_spawn.wav
        
        if self.wave_aliens_to_spawn == 0:
            self.wave_spawning = False

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            
            # Find a new target if needed
            if tower['target'] is None or tower['target'] not in self.aliens:
                tower['target'] = None
                potential_targets = [
                    a for a in self.aliens 
                    if self._dist_sq(tower['pixel_pos'], a['pos']) < tower['spec']['range']**2
                ]
                if potential_targets:
                    # Target alien furthest along the path
                    tower['target'] = max(potential_targets, key=lambda a: a['waypoint_idx'])

            # Fire if ready and has a target
            if tower['target'] and tower['cooldown'] <= 0:
                target_pos = tower['target']['pos']
                if self._dist_sq(tower['pixel_pos'], target_pos) > tower['spec']['range']**2:
                    tower['target'] = None # Target out of range
                else:
                    tower['cooldown'] = tower['spec']['fire_rate']
                    # Cannon
                    if tower['spec']['name'] == "Cannon":
                        # Sound: cannon_fire.wav
                        self.projectiles.append({
                            "pos": list(tower['pixel_pos']),
                            "target": tower['target'],
                            "spec": tower['spec']
                        })
                    # Laser
                    elif tower['spec']['name'] == "Laser":
                        # Sound: laser_beam.wav
                        tower['target']['health'] -= tower['spec']['damage']
                        self.step_reward += 0.1
                        self._create_particles(target_pos, self.TOWER_SPECS[1]['color'], 1, 1, 5)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj['target']
            if target not in self.aliens:
                self.projectiles.remove(proj)
                continue

            direction = np.array(target['pos']) - np.array(proj['pos'])
            dist = np.linalg.norm(direction)
            if dist < proj['spec']['proj_speed']:
                # Hit
                target['health'] -= proj['spec']['damage']
                self.score += 1
                self.step_reward += 0.1
                # Sound: projectile_hit.wav
                self._create_particles(target['pos'], self.TOWER_SPECS[0]['color'], 5, 2, 10)
                self.projectiles.remove(proj)
            else:
                # Move
                direction = direction / dist
                proj['pos'][0] += direction[0] * proj['spec']['proj_speed']
                proj['pos'][1] += direction[1] * proj['spec']['proj_speed']

    def _update_aliens(self):
        base_pixel_pos = self._grid_to_pixel(*self.base_pos_grid)
        is_base_under_attack = False

        for alien in self.aliens[:]:
            if alien['health'] <= 0:
                # Sound: alien_die.wav
                self.score += 10
                self._create_particles(alien['pos'], (255, 50, 50), 20, 3, 20)
                self.aliens.remove(alien)
                continue

            if alien['waypoint_idx'] >= len(self.alien_waypoints):
                # Reached base
                self.base_health -= alien['health'] / alien['max_health'] * 10 # Damage scaled by remaining health
                self.base_health = max(0, self.base_health)
                # Sound: base_damage.wav
                self._create_particles(alien['pos'], (255, 100, 0), 15, 4, 25)
                self.aliens.remove(alien)
                continue

            target_pos = self.alien_waypoints[alien['waypoint_idx']]
            direction = np.array(target_pos) - np.array(alien['pos'])
            dist = np.linalg.norm(direction)
            
            if dist < alien['speed']:
                alien['waypoint_idx'] += 1
            else:
                direction = direction / dist
                alien['pos'][0] += direction[0] * alien['speed']
                alien['pos'][1] += direction[1] * alien['speed']

        if any(self._dist_sq(a['pos'], base_pixel_pos) < self.GRID_SIZE**2 for a in self.aliens):
             is_base_under_attack = True
        
        if is_base_under_attack:
            self.step_reward -= 0.01

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.win:
            self.game_over = True
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                if (x, y) in self.path_grid_coords:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw base
        base_px, base_py = self._grid_to_pixel(*self.base_pos_grid)
        base_rect = pygame.Rect(0, 0, self.GRID_SIZE - 4, self.GRID_SIZE - 4)
        base_rect.center = (base_px, base_py)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BASE_STROKE, base_rect, 2, border_radius=4)

        # Draw towers
        for tower in self.towers:
            pos = tower['pixel_pos']
            spec = tower['spec']
            if spec['name'] == 'Cannon':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, spec['color'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, (255,255,255))
            elif spec['name'] == 'Laser':
                side = 16
                rect = pygame.Rect(pos[0] - side/2, pos[1] - side/2, side, side)
                pygame.draw.rect(self.screen, spec['color'], rect, border_radius=3)
                pygame.draw.rect(self.screen, (255,255,255), rect, 1, border_radius=3)
                # Draw laser beam if firing
                if tower['target'] and tower['cooldown'] > spec['fire_rate'] - 2: # Show beam for 2 frames
                    pygame.draw.aaline(self.screen, spec['color'], pos, tower['target']['pos'], 2)
        
        # Draw aliens
        for alien in self.aliens:
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            size = 8
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, (255, 50, 50), rect)
            pygame.draw.rect(self.screen, (255, 150, 150), rect, 2)

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj['spec']['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255,255,255))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw cursor
        self._render_cursor()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_main.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Base Health Bar
        bar_w, bar_h = 100, 15
        bar_x, bar_y = self._grid_to_pixel(*self.base_pos_grid)[0] - bar_w // 2, self._grid_to_pixel(*self.base_pos_grid)[1] - self.GRID_SIZE
        health_ratio = self.base_health / self.STARTING_BASE_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_ratio), bar_h), border_radius=3)
        
        # Selected Tower UI
        st_x, st_y = 10, self.HEIGHT - 40
        spec = self.TOWER_SPECS[self.selected_tower_type]
        name_text = self.font_main.render(f"Selected: {spec['name']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (st_x, st_y))

        # Game Over / Win Text
        if self.game_over:
            outcome_text = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 50, 50)
            text_surf = self.font_large.render(outcome_text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(text_surf, text_rect)

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        px, py = self._grid_to_pixel(gx, gy)
        
        is_on_path = (gx, gy) in self.path_grid_coords
        is_on_base = (gx, gy) == self.base_pos_grid
        is_occupied = any(t['grid_pos'] == [gx, gy] for t in self.towers)
        is_valid = not is_on_path and not is_occupied and not is_on_base

        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        cursor_surf = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, color, (0, 0, self.GRID_SIZE, self.GRID_SIZE), 0, border_radius=3)
        self.screen.blit(cursor_surf, (gx * self.GRID_SIZE, gy * self.GRID_SIZE))
        
        # Draw range indicator
        spec = self.TOWER_SPECS[self.selected_tower_type]
        pygame.gfxdraw.aacircle(self.screen, px, py, spec['range'], (*color[:3], 100))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "base_health": self.base_health}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Helper Functions ---
    def _grid_to_pixel(self, gx, gy):
        return int((gx + 0.5) * self.GRID_SIZE), int((gy + 0.5) * self.GRID_SIZE)

    def _dist_sq(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def _create_particles(self, pos, color, count, size, life):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life + self.np_random.integers(0, life // 2),
                'max_life': life * 1.5,
                'size': self.np_random.integers(1, size+1),
                'color': color
            })