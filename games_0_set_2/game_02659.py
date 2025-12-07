import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor, Shift to cycle tower type, Space to place tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on the grid."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60 * 5  # 5 minutes max

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_GRID_HOVER = (60, 80, 100)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (40, 200, 40)
        self.COLOR_HEALTH_RED = (200, 40, 40)
        self.PARTICLE_COLORS = [(255, 255, 100), (255, 150, 50), (255, 100, 50)]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)

        # --- Game Data ---
        self._define_game_data()

        # Initialize state variables to allow for validation call before reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_reward_given = False

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 120

        self.current_wave = 0
        self.wave_timer = 150
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._determine_buildable_grid()
        
        self.validate_implementation()

    def _define_game_data(self):
        # Path definition
        self.PATH = [
            (-20, 100), (150, 100), (150, 300), (450, 300), (450, 50), (self.WIDTH + 20, 50)
        ]
        self.path_segments = []
        for i in range(len(self.PATH) - 1):
            p1 = pygame.Vector2(self.PATH[i])
            p2 = pygame.Vector2(self.PATH[i+1])
            self.path_segments.append({
                'start': p1,
                'end': p2,
                'length': p1.distance_to(p2),
                'direction': (p2 - p1).normalize() if p1.distance_to(p2) > 0 else pygame.Vector2(0,0)
            })
        self.total_path_length = sum(s['length'] for s in self.path_segments)
        
        # Tower placement grid
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        # Tower specifications
        self.TOWER_SPECS = {
            "Machine Gun": {"cost": 40, "range": 80, "damage": 2.5, "fire_rate": 5, "color": (50, 150, 255), "shape": "rect"},
            "Cannon": {"cost": 100, "range": 120, "damage": 20, "fire_rate": 0.8, "color": (255, 200, 50), "shape": "circle"},
            "Slower": {"cost": 75, "range": 70, "damage": 0.5, "fire_rate": 2, "slow": 0.5, "slow_duration": 2 * self.FPS, "color": (200, 50, 255), "shape": "poly"},
        }
        self.TOWER_TYPES = list(self.TOWER_SPECS.keys())

        # Wave configuration
        self.WAVE_CONFIG = [
            {'count': 8, 'delay': 45}, {'count': 12, 'delay': 40}, {'count': 15, 'delay': 35},
            {'count': 20, 'delay': 30}, {'count': 25, 'delay': 25}, {'count': 30, 'delay': 20},
            {'count': 35, 'delay': 18}, {'count': 40, 'delay': 15}, {'count': 45, 'delay': 12},
            {'count': 60, 'delay': 10}
        ]
        self.TOTAL_WAVES = len(self.WAVE_CONFIG)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_reward_given = False

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 120

        self.current_wave = 0
        self.wave_timer = 150  # Initial delay before first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._determine_buildable_grid()

        return self._get_observation(), self._get_info()

    def _determine_buildable_grid(self):
        self.buildable_grid = np.ones((self.GRID_W, self.GRID_H), dtype=bool)
        path_width = 20
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                cell_center = pygame.Vector2(x * self.GRID_SIZE + self.GRID_SIZE / 2, y * self.GRID_SIZE + self.GRID_SIZE / 2)
                for i in range(len(self.PATH) - 1):
                    p1 = pygame.Vector2(self.PATH[i])
                    p2 = pygame.Vector2(self.PATH[i+1])
                    # Point-segment distance check
                    l2 = p1.distance_squared_to(p2)
                    if l2 == 0.0:
                        dist_sq = cell_center.distance_squared_to(p1)
                    else:
                        t = max(0, min(1, (cell_center - p1).dot(p2 - p1) / l2))
                        projection = p1 + t * (p2 - p1)
                        dist_sq = cell_center.distance_squared_to(projection)
                    
                    if dist_sq < path_width**2:
                        self.buildable_grid[x, y] = False
                        break
        # Make base area unbuildable
        base_grid_x, base_grid_y = int(self.PATH[-1][0] // self.GRID_SIZE), int(self.PATH[-1][1] // self.GRID_SIZE)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= base_grid_x + i < self.GRID_W and 0 <= base_grid_y + j < self.GRID_H:
                    self.buildable_grid[base_grid_x + i, base_grid_y + j] = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Handle player input
        reward += self._handle_input(action)
        
        # Update game state
        reward += self._update_waves()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # Check for termination conditions
        terminated = False
        if (self.base_health <= 0 or self.steps >= self.MAX_STEPS) and not self.terminal_reward_given:
            terminated = True
            reward += -100
            self.game_over = True
            self.terminal_reward_given = True
        elif self.current_wave >= self.TOTAL_WAVES and not self.enemies and not self.enemies_to_spawn and not self.terminal_reward_given:
            terminated = True
            reward += 100
            self.game_over = True
            self.terminal_reward_given = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)
        
        # --- Cycle Tower Type (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle.wav

        # --- Place Tower (on press) ---
        if space_held and not self.prev_space_held:
            tower_type = self.TOWER_TYPES[self.selected_tower_idx]
            spec = self.TOWER_SPECS[tower_type]
            grid_x, grid_y = self.cursor_pos
            
            is_buildable = self.buildable_grid[grid_x, grid_y]
            is_occupied = any(t['grid_pos'] == [grid_x, grid_y] for t in self.towers)
            
            if self.resources >= spec['cost'] and is_buildable and not is_occupied:
                self.resources -= spec['cost']
                pos = (grid_x * self.GRID_SIZE + self.GRID_SIZE / 2, grid_y * self.GRID_SIZE + self.GRID_SIZE / 2)
                self.towers.append({
                    "type": tower_type,
                    "pos": pygame.Vector2(pos),
                    "grid_pos": [grid_x, grid_y],
                    "cooldown": 0,
                    "target": None
                })
                # sfx: place_tower.wav
        
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        return reward

    def _update_waves(self):
        if self.current_wave >= self.TOTAL_WAVES:
            return 0
        
        if not self.enemies and self.enemies_to_spawn == 0:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                wave_data = self.WAVE_CONFIG[self.current_wave - 1]
                self.enemies_to_spawn = wave_data['count']
                self.spawn_timer = 0
                # sfx: wave_start.wav

        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.spawn_timer = self.WAVE_CONFIG[self.current_wave - 1]['delay']
        return 0

    def _spawn_enemy(self):
        health = 15 * (1.15 ** self.current_wave)
        speed = 25 * (1.05 ** self.current_wave) / self.FPS
        value = 2 + self.current_wave
        self.enemies.append({
            "pos": pygame.Vector2(self.PATH[0]),
            "path_dist": 0.0,
            "path_segment": 0,
            "health": health,
            "max_health": health,
            "speed": speed,
            "value": value,
            "slow_timer": 0
        })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            current_speed = enemy['speed']
            if enemy['slow_timer'] > 0:
                current_speed *= enemy['slow_effect']
                enemy['slow_timer'] -= 1

            enemy['path_dist'] += current_speed
            
            dist_traveled = 0
            new_pos = None
            for segment in self.path_segments:
                if dist_traveled + segment['length'] >= enemy['path_dist']:
                    progress_on_segment = enemy['path_dist'] - dist_traveled
                    new_pos = segment['start'] + segment['direction'] * progress_on_segment
                    break
                dist_traveled += segment['length']

            if new_pos:
                enemy['pos'] = new_pos
            else: # Reached the end
                self.base_health -= enemy['health'] # Damage based on remaining health
                self.base_health = max(0, self.base_health)
                reward -= enemy['health']
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                self.particles.append({'pos': pygame.Vector2(self.PATH[-1]), 'radius': 15, 'max_radius': 40, 'color': self.COLOR_HEALTH_RED, 'type': 'shockwave'})
        return reward

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            # Find a new target if needed
            if tower['target'] is None or tower['target'] not in self.enemies or tower['pos'].distance_to(tower['target']['pos']) > spec['range']:
                tower['target'] = None
                possible_targets = [e for e in self.enemies if tower['pos'].distance_to(e['pos']) <= spec['range']]
                if possible_targets:
                    # Target enemy furthest along the path
                    tower['target'] = max(possible_targets, key=lambda e: e['path_dist'])

            # Fire if ready and has a target
            if tower['cooldown'] == 0 and tower['target']:
                self.projectiles.append({
                    "pos": tower['pos'].copy(),
                    "target": tower['target'],
                    "speed": 15,
                    "damage": spec['damage'],
                    "slow": spec.get('slow', 0),
                    "slow_duration": spec.get('slow_duration', 0),
                    "color": spec['color']
                })
                tower['cooldown'] = self.FPS / spec['fire_rate']
                # sfx: shoot_laser.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = (proj['target']['pos'] - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_to(proj['target']['pos']) < 10:
                enemy = proj['target']
                enemy['health'] -= proj['damage']
                reward += 0.1 # Reward for hitting
                
                if proj['slow'] > 0:
                    enemy['slow_timer'] = proj['slow_duration']
                    enemy['slow_effect'] = 1.0 - proj['slow']

                # sfx: hit_enemy.wav
                self.particles.append({'pos': enemy['pos'].copy(), 'radius': 0, 'max_radius': 10, 'color': proj['color'], 'type': 'burst'})
                
                if enemy['health'] <= 0:
                    reward += 1.0 # Reward for kill
                    self.resources += enemy['value']
                    if enemy in self.enemies:
                        self.enemies.remove(enemy)
                    # sfx: enemy_die.wav
                    self.particles.append({'pos': enemy['pos'].copy(), 'radius': 0, 'max_radius': 20, 'color': self.COLOR_HEALTH_RED, 'type': 'burst'})
                
                self.projectiles.remove(proj)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            if p['type'] == 'burst':
                p['radius'] += 2
                if p['radius'] >= p['max_radius']:
                    self.particles.remove(p)
            elif p['type'] == 'shockwave':
                p['radius'] += 1
                if p['radius'] >= p['max_radius']:
                    self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH, 30)
        
        # Draw buildable grid and cursor
        cursor_world_pos = (self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.buildable_grid[x, y]:
                    rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor and tower range preview
        selected_tower_type = self.TOWER_TYPES[self.selected_tower_idx]
        spec = self.TOWER_SPECS[selected_tower_type]
        cursor_center_pos = (cursor_world_pos[0] + self.GRID_SIZE/2, cursor_world_pos[1] + self.GRID_SIZE/2)
        
        can_build = self.resources >= spec['cost'] and \
                    self.buildable_grid[self.cursor_pos[0], self.cursor_pos[1]] and \
                    not any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        
        range_color = (0, 255, 0, 50) if can_build else (255, 0, 0, 50)
        s = pygame.Surface((spec['range']*2, spec['range']*2), pygame.SRCALPHA)
        pygame.draw.circle(s, range_color, (spec['range'], spec['range']), spec['range'])
        self.screen.blit(s, (cursor_center_pos[0] - spec['range'], cursor_center_pos[1] - spec['range']))
        
        cursor_rect = pygame.Rect(cursor_world_pos[0], cursor_world_pos[1], self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_HOVER, cursor_rect, 2)

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            if spec['shape'] == 'rect':
                pygame.draw.rect(self.screen, spec['color'], (pos[0]-10, pos[1]-10, 20, 20))
            elif spec['shape'] == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, spec['color'])
            elif spec['shape'] == 'poly':
                points = [(pos[0], pos[1]-12), (pos[0]-10, pos[1]+8), (pos[0]+10, pos[1]+8)]
                pygame.gfxdraw.filled_polygon(self.screen, points, spec['color'])

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            color = (255, 100, 100) if enemy['slow_timer'] <= 0 else (100, 100, 255)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, (255,255,255))
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 14
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos[0]-bar_w/2, pos[1]-15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0]-bar_w/2, pos[1]-15, bar_w * health_pct, 3))
            
        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.rect(self.screen, proj['color'], (pos[0]-2, pos[1]-2, 4, 4))
        
        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            if p['type'] == 'burst':
                alpha = 255 * (1 - p['radius'] / p['max_radius'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), (*p['color'], alpha))
            elif p['type'] == 'shockwave':
                alpha = 150 * (1 - p['radius'] / p['max_radius'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), (*p['color'], alpha))
        
        # Draw base
        base_pos = (int(self.PATH[-1][0]), int(self.PATH[-1][1]))
        pygame.gfxdraw.box(self.screen, (base_pos[0]-15, base_pos[1]-15, 30, 30), (*self.COLOR_BASE, 180))
        pygame.gfxdraw.rectangle(self.screen, (base_pos[0]-15, base_pos[1]-15, 30, 30), self.COLOR_BASE)
        # Base health bar
        health_pct = self.base_health / self.max_base_health
        bar_w = 50
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (base_pos[0]-bar_w/2, base_pos[1]-25, bar_w, 5))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (base_pos[0]-bar_w/2, base_pos[1]-25, bar_w * health_pct, 5))

    def _render_ui(self):
        # Wave info
        wave_text = f"Wave: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.wave_timer > 0 and self.current_wave < self.TOTAL_WAVES:
            wave_text += f" (Next in {self.wave_timer/self.FPS:.1f}s)"
        surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(surf, (10, 10))

        # Resources
        res_text = f"Resources: {self.resources}"
        surf = self.font_small.render(res_text, True, (255, 220, 100))
        self.screen.blit(surf, (self.WIDTH - surf.get_width() - 10, 10))

        # Selected Tower
        tower_type = self.TOWER_TYPES[self.selected_tower_idx]
        spec = self.TOWER_SPECS[tower_type]
        tower_text = f"Selected: {tower_type} (Cost: {spec['cost']})"
        color = self.COLOR_TEXT if self.resources >= spec['cost'] else self.COLOR_HEALTH_RED
        surf = self.font_small.render(tower_text, True, color)
        self.screen.blit(surf, (10, self.HEIGHT - surf.get_height() - 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            color = self.COLOR_HEALTH_GREEN if self.base_health > 0 else self.COLOR_HEALTH_RED
            surf = self.font_large.render(msg, True, color)
            pos = (self.WIDTH/2 - surf.get_width()/2, self.HEIGHT/2 - surf.get_height()/2)
            self.screen.blit(surf, pos)

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
        
        # print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()