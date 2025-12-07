
# Generated: 2025-08-28T04:02:30.332004
# Source Brief: brief_05128.md
# Brief Index: 5128

        
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
        "Controls: Use arrow keys to move the placement cursor. Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from incoming enemy waves by strategically placing attack towers on an isometric grid. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (44, 62, 80)
    COLOR_PATH = (52, 73, 94)
    COLOR_GRID = (62, 83, 104)
    
    COLOR_BASE = (52, 152, 219)
    COLOR_BASE_DMG = (231, 76, 60)

    COLOR_ENEMY = (231, 76, 60)
    COLOR_ENEMY_OUTLINE = (192, 57, 43)
    
    COLOR_TOWER_CANNON = (46, 204, 113)
    COLOR_TOWER_LASER = (155, 89, 182)
    
    COLOR_PROJ_CANNON = (241, 196, 15)
    COLOR_PROJ_LASER = (255, 121, 233)

    COLOR_CURSOR = (241, 196, 15)
    COLOR_CURSOR_INVALID = (192, 57, 43)

    COLOR_PLACEABLE_TILE = (22, 160, 133, 100)

    COLOR_UI_TEXT = (236, 240, 241)
    COLOR_UI_SUCCESS = (46, 204, 113)
    COLOR_UI_FAILURE = (231, 76, 60)
    COLOR_HEALTH_BAR_BG = (90, 90, 90)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- Game Config ---
        self.iso_tile_width = 32
        self.iso_tile_height = 16
        self.grid_size_x = 14
        self.grid_size_y = 14
        self.origin_x = self.width // 2
        self.origin_y = 80
        
        self.max_steps = 30 * 180 # 3 minutes at 30fps
        self.max_waves = 10
        self.wave_cooldown_time = 30 * 5 # 5 seconds
        
        self.enemy_path = [
            (0, 7), (1, 7), (2, 7), (2, 6), (2, 5), (2, 4), (3, 4), (4, 4), 
            (5, 4), (5, 5), (5, 6), (6, 6), (7, 6), (8, 6), (8, 5), (8, 4), 
            (8, 3), (8, 2), (9, 2), (10, 2), (11, 2), (11, 3), (11, 4), 
            (11, 5), (11, 6), (11, 7), (11, 8), (10, 8), (9, 8), (8, 8), (7, 8)
        ]
        self.base_pos_grid = (7, 8)
        
        self.placeable_tiles = set()
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if (x, y) not in self.enemy_path and (x, y) != self.base_pos_grid:
                    self.placeable_tiles.add((x, y))

        self.tower_types = [
            {"name": "Cannon", "cost": 50, "range": 80, "damage": 25, "fire_rate": 1.0, "color": self.COLOR_TOWER_CANNON, "proj_color": self.COLOR_PROJ_CANNON, "proj_speed": 5},
            {"name": "Laser", "cost": 75, "range": 120, "damage": 8, "fire_rate": 0.2, "color": self.COLOR_TOWER_LASER, "proj_color": self.COLOR_PROJ_LASER, "proj_speed": 15},
        ]

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 75
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.grid_size_x // 2, self.grid_size_y // 2]
        self.selected_tower_idx = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.current_wave_num = 0
        self.wave_in_progress = False
        self.enemies_spawned_this_wave = 0
        self.wave_cooldown = self.wave_cooldown_time // 2
        
        self.last_reward = 0.0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for surviving
        self.last_reward = 0.0

        if not self.game_over:
            # --- Handle Input ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            space_pressed = space_held and not self.prev_space_held
            shift_pressed = shift_held and not self.prev_shift_held

            self._handle_input(movement, space_pressed, shift_pressed)
            
            self.prev_space_held, self.prev_shift_held = space_held, shift_held

            # --- Game Logic ---
            self._update_wave_manager()
            reward += self._update_towers()
            self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
        
        self.score += reward
        self.last_reward = reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.win:
            reward -= 10.0 # Penalty for losing
            self.last_reward -= 10.0
        elif terminated and self.win:
            reward += 10.0 # Reward for winning
            self.last_reward += 10.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_size_x - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_size_y - 1)

        # Cycle tower
        if shift_pressed:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
            # sfx: UI_Cycle

        # Place tower
        if space_pressed:
            pos = tuple(self.cursor_pos)
            tower_def = self.tower_types[self.selected_tower_idx]
            can_afford = self.resources >= tower_def["cost"]
            is_placeable = pos in self.placeable_tiles
            is_occupied = any(t['pos'] == pos for t in self.towers)

            if can_afford and is_placeable and not is_occupied:
                self.resources -= tower_def["cost"]
                new_tower = {
                    "pos": pos,
                    "type": tower_def,
                    "cooldown": 0,
                }
                self.towers.append(new_tower)
                # sfx: Build_Tower

    def _update_wave_manager(self):
        if self.win: return
        
        if not self.wave_in_progress:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.current_wave_num < self.max_waves:
                self.current_wave_num += 1
                self.wave_in_progress = True
                self.enemies_spawned_this_wave = 0
        else: # Wave is in progress
            wave_def = self._get_wave_definition(self.current_wave_num)
            if self.enemies_spawned_this_wave < wave_def['count']:
                if self.steps % wave_def['spawn_rate'] == 0:
                    self._spawn_enemy(wave_def)
                    self.enemies_spawned_this_wave += 1
            elif not self.enemies: # Wave is over
                self.wave_in_progress = False
                self.wave_cooldown = self.wave_cooldown_time
                self.last_reward += 1.0 # Wave complete reward
                self.score += 1.0
                # sfx: Wave_Complete
                if self.current_wave_num >= self.max_waves:
                    self.win = True
                    self.game_over = True

    def _get_wave_definition(self, wave_num):
        return {
            'count': 3 + wave_num * 2,
            'health': 50 + wave_num * 15,
            'speed': 0.8 + wave_num * 0.05,
            'spawn_rate': max(5, 30 - wave_num * 2),
            'value': 5 + wave_num,
        }

    def _spawn_enemy(self, wave_def):
        start_pos = self._grid_to_world(self.enemy_path[0][0], self.enemy_path[0][1])
        new_enemy = {
            "pos": list(start_pos),
            "health": wave_def['health'],
            "max_health": wave_def['health'],
            "speed": wave_def['speed'],
            "path_idx": 0,
            "value": wave_def['value'],
        }
        self.enemies.append(new_enemy)

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            target = None
            min_dist = tower['type']['range'] ** 2
            tower_pos = self._grid_to_world(tower['pos'][0], tower['pos'][1])

            for enemy in self.enemies:
                dist_sq = (enemy['pos'][0] - tower_pos[0])**2 + (enemy['pos'][1] - tower_pos[1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower['cooldown'] = int(30 * tower['type']['fire_rate'])
                self.projectiles.append({
                    "pos": list(tower_pos),
                    "target": target,
                    "type": tower['type'],
                })
                # sfx: Tower_Shoot

        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj['target']
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            dx = target['pos'][0] - proj['pos'][0]
            dy = target['pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['type']['proj_speed']:
                target['health'] -= proj['type']['damage']
                self._create_particles(target['pos'], proj['type']['proj_color'], 5)
                # sfx: Enemy_Hit
                self.projectiles.remove(proj)
            else:
                proj['pos'][0] += (dx / dist) * proj['type']['proj_speed']
                proj['pos'][1] += (dy / dist) * proj['type']['proj_speed']

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Check if dead
            if enemy['health'] <= 0:
                reward += 0.1 # Kill reward
                self.resources += enemy['value']
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 2.0)
                # sfx: Enemy_Destroyed
                self.enemies.remove(enemy)
                continue

            # Check if reached base
            if enemy['path_idx'] >= len(self.enemy_path) - 1:
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                reward -= 1.0 # Base damage penalty
                self._create_particles(enemy['pos'], self.COLOR_BASE_DMG, 20, 3.0)
                # sfx: Base_Damage
                self.enemies.remove(enemy)
                continue
            
            # Move along path
            target_node_grid = self.enemy_path[enemy['path_idx'] + 1]
            target_pos_world = self._grid_to_world(target_node_grid[0], target_node_grid[1])
            
            dx = target_pos_world[0] - enemy['pos'][0]
            dy = target_pos_world[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['path_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['size'] *= 0.95

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.max_steps:
            self.game_over = True
            return True
        return False

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
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave_num,
            "win": self.win,
        }

    # --- Rendering Methods ---
    def _grid_to_world(self, x, y):
        screen_x = self.origin_x + (x - y) * self.iso_tile_width / 2
        screen_y = self.origin_y + (x + y) * self.iso_tile_height / 2
        return screen_x, screen_y

    def _draw_iso_rect(self, surface, color, grid_pos, height_offset=0):
        x, y = grid_pos
        w, h = self.iso_tile_width, self.iso_tile_height
        x_w, y_w = self._grid_to_world(x, y)
        y_w -= height_offset
        
        points = [
            (x_w, y_w),
            (x_w + w / 2, y_w + h / 2),
            (x_w, y_w + h),
            (x_w - w / 2, y_w + h / 2)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        
    def _render_grid_and_path(self):
        # Draw grid lines
        for i in range(self.grid_size_x + 1):
            start = self._grid_to_world(i, 0)
            end = self._grid_to_world(i, self.grid_size_y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.grid_size_y + 1):
            start = self._grid_to_world(0, i)
            end = self._grid_to_world(self.grid_size_x, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        
        # Draw path
        for pos in self.enemy_path:
            self._draw_iso_rect(self.screen, self.COLOR_PATH, pos)
            
        # Draw placeable tiles
        for pos in self.placeable_tiles:
            x_w, y_w = self._grid_to_world(pos[0], pos[1])
            points = [
                (x_w, y_w), (x_w + self.iso_tile_width / 2, y_w + self.iso_tile_height / 2),
                (x_w, y_w + self.iso_tile_height), (x_w - self.iso_tile_width / 2, y_w + self.iso_tile_height / 2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLACEABLE_TILE)

    def _render_base(self):
        self._draw_iso_rect(self.screen, self.COLOR_BASE, self.base_pos_grid, height_offset=8)
        self._draw_iso_rect(self.screen, self.COLOR_BASE, self.base_pos_grid, height_offset=4)
        self._draw_iso_rect(self.screen, self.COLOR_BASE, self.base_pos_grid)
        
        # Health bar
        base_world_pos = self._grid_to_world(self.base_pos_grid[0], self.base_pos_grid[1])
        bar_w = self.iso_tile_width
        bar_h = 5
        bar_x = base_world_pos[0] - bar_w / 2
        bar_y = base_world_pos[1] - 15
        
        health_ratio = self.base_health / self.max_base_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_SUCCESS, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_towers(self):
        for tower in self.towers:
            tower_def = tower['type']
            self._draw_iso_rect(self.screen, tower_def['color'], tower['pos'], height_offset=4)
            # Tower range indicator
            if tuple(self.cursor_pos) == tower['pos']:
                pos_w = self._grid_to_world(tower['pos'][0], tower['pos'][1])
                pygame.gfxdraw.aacircle(self.screen, int(pos_w[0]), int(pos_w[1]), tower_def['range'], (255, 255, 255, 50))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_x, pos_y = int(enemy['pos'][0]), int(enemy['pos'][1])
            size = 6
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, size, self.COLOR_ENEMY_OUTLINE)
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 12
            bar_h = 3
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos_x - bar_w/2, pos_y - size - 5, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_UI_SUCCESS, (pos_x - bar_w/2, pos_y - size - 5, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            if proj['type']['name'] == 'Laser':
                pygame.draw.aaline(self.screen, proj['type']['proj_color'], proj['pos'], proj['target']['pos'], 1)
            else: # Cannon
                pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['type']['proj_color'])
                pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, (255,255,255,150))

    def _render_particles(self):
        for p in self.particles:
            size = int(p['size'])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, p['color'])

    def _render_cursor(self):
        pos = tuple(self.cursor_pos)
        tower_def = self.tower_types[self.selected_tower_idx]
        can_afford = self.resources >= tower_def["cost"]
        is_placeable = pos in self.placeable_tiles
        is_occupied = any(t['pos'] == pos for t in self.towers)
        
        color = self.COLOR_CURSOR if (can_afford and is_placeable and not is_occupied) else self.COLOR_CURSOR_INVALID
        
        self._draw_iso_rect(self.screen, (*color, 100), pos)
        
        x_w, y_w = self._grid_to_world(pos[0], pos[1])
        w, h = self.iso_tile_width, self.iso_tile_height
        points = [
            (x_w, y_w), (x_w + w / 2, y_w + h / 2),
            (x_w, y_w + h), (x_w - w / 2, y_w + h / 2)
        ]
        pygame.draw.aalines(self.screen, color, True, points)

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, (0,0,0,100), (0, 0, self.width, 30))
        
        # Resources
        res_text = self.font_small.render(f"RESOURCES: ${self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 7))
        
        # Wave
        wave_str = f"WAVE: {self.current_wave_num}/{self.max_waves}"
        if not self.wave_in_progress and not self.win:
            wave_str += f" (Next in {self.wave_cooldown/30:.1f}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.width // 2 - wave_text.get_width() // 2, 7))

        # Selected Tower
        tower_def = self.tower_types[self.selected_tower_idx]
        tower_text = self.font_small.render(f"BUILD: {tower_def['name']} (${tower_def['cost']})", True, tower_def['color'])
        self.screen.blit(tower_text, (self.width - tower_text.get_width() - 10, 7))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "VICTORY!"
                color = self.COLOR_UI_SUCCESS
            else:
                msg = "BASE DESTROYED"
                color = self.COLOR_UI_FAILURE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': random.randint(10, 20),
                'color': color,
                'size': random.uniform(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0] 

    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)

    while running:
        action = [0, 0, 0] # Reset actions each frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()