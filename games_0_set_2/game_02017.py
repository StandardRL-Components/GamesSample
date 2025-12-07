
# Generated: 2025-08-28T03:24:31.847176
# Source Brief: brief_02017.md
# Brief Index: 2017

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrows to move cursor. Space to place a blue turret, Shift to place a yellow turret."
    )
    game_description = (
        "Defend your base from waves of geometric enemies by placing turrets on an isometric grid."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PATH = (50, 58, 80)
        self.COLOR_BASE = (50, 180, 120)
        self.COLOR_BASE_DMG = (200, 80, 80)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TOWER_1 = (60, 120, 255)
        self.COLOR_TOWER_2 = (255, 200, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)

        # Game path and grid setup
        self.path_coords = self._generate_path()
        self.placement_tiles = self._generate_placement_tiles()

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_active = False
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0,0]
        self.last_base_hit_time = 0
        
        self.reset()
        self.validate_implementation()

    def _generate_path(self):
        path = []
        for i in range(10): path.append((i, 10))
        for i in range(10, 2, -1): path.append((9, i))
        for i in range(9, -1, -1): path.append((i, 3))
        return path

    def _generate_placement_tiles(self):
        tiles = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.path_coords:
                    tiles.add((x, y))
        # Remove tiles near base
        tiles.discard((0,3))
        tiles.discard((1,3))
        tiles.discard((0,4))
        return tiles

    def _iso_to_screen(self, gx, gy):
        return (
            self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH_HALF,
            self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT_HALF
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 100
        self.resources = 50
        self.current_wave = 0
        self.wave_active = False
        self.wave_timer = 150  # 5 seconds at 30fps
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()

        self.steps += 1
        
        # 1. Handle player input
        self._handle_player_input(action)

        # 2. Update game state
        self._update_wave_management()
        reward += self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # 3. Check for wave completion
        if self.wave_active and not self.enemies:
            self.wave_active = False
            self.wave_timer = 240 # 8 seconds
            if self.current_wave > 0:
                reward += 5.0
                self.score += 50
                self.resources += 20 + self.current_wave * 5

        # 4. Check for termination
        terminated = False
        if self.base_health <= 0:
            reward = -100.0
            self.game_over = True
            terminated = True
        elif self.current_wave == 5 and not self.wave_active and not self.enemies:
            reward = 100.0
            self.game_over = True
            self.win = True
            terminated = True
        elif self.steps >= 5000: # Max episode length
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, action):
        movement, place_t1, place_t2 = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place towers
        cursor_tuple = tuple(self.cursor_pos)
        is_occupied = any(t['grid_pos'] == cursor_tuple for t in self.towers)
        
        if place_t1 and cursor_tuple in self.placement_tiles and not is_occupied and self.resources >= 10:
            self.resources -= 10
            self.towers.append({'type': 1, 'grid_pos': cursor_tuple, 'cooldown': 0, 'range': 4, 'damage': 12, 'fire_rate': 20})
            self._create_particles(self._iso_to_screen(*cursor_tuple), 20, self.COLOR_TOWER_1)
        elif place_t2 and cursor_tuple in self.placement_tiles and not is_occupied and self.resources >= 20:
            self.resources -= 20
            self.towers.append({'type': 2, 'grid_pos': cursor_tuple, 'cooldown': 0, 'range': 7, 'damage': 8, 'fire_rate': 45})
            self._create_particles(self._iso_to_screen(*cursor_tuple), 20, self.COLOR_TOWER_2)

    def _update_wave_management(self):
        if not self.wave_active and not self.win:
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave < 5:
                self.current_wave += 1
                self.wave_active = True
                num_enemies = 5 + (self.current_wave - 1) * 2
                enemy_speed = 0.02 + (self.current_wave - 1) * 0.005
                enemy_health = 20 + (self.current_wave - 1) * 10
                
                for i in range(num_enemies):
                    spawn_pos = self._iso_to_screen(self.path_coords[0][0], self.path_coords[0][1] - 2)
                    offset_x = self.np_random.uniform(-10, 10)
                    offset_y = self.np_random.uniform(-10, 10)
                    self.enemies.append({
                        'pos': [spawn_pos[0] + offset_x, spawn_pos[1] + offset_y - i * 15],
                        'health': enemy_health, 'max_health': enemy_health,
                        'speed': enemy_speed, 'path_index': 0, 'id': self.np_random.random()
                    })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            target = None
            min_dist = tower['range'] * self.TILE_WIDTH_HALF
            tower_screen_pos = self._iso_to_screen(*tower['grid_pos'])
            
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower_screen_pos[0], enemy['pos'][1] - tower_screen_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['fire_rate']
                color = self.COLOR_TOWER_1 if tower['type'] == 1 else self.COLOR_TOWER_2
                self.projectiles.append({
                    'start_pos': list(tower_screen_pos),
                    'pos': list(tower_screen_pos),
                    'target_id': target['id'],
                    'damage': tower['damage'],
                    'speed': 8,
                    'color': color
                })
                # Muzzle flash
                self._create_particles(tower_screen_pos, 5, color, 0.5, 5)
        return reward

    def _update_projectiles(self):
        projectiles_to_keep = []
        for p in self.projectiles:
            target = next((e for e in self.enemies if e['id'] == p['target_id']), None)
            
            if not target: # Target is gone
                p['pos'][0] += p.get('vx', 0) * p['speed']
                p['pos'][1] += p.get('vy', 0) * p['speed']
                if not self.screen.get_rect().collidepoint(p['pos']):
                    continue # Fizzle out off-screen
            else:
                target_pos = target['pos']
                dx = target_pos[0] - p['pos'][0]
                dy = target_pos[1] - p['pos'][1]
                dist = math.hypot(dx, dy)

                if dist < p['speed']:
                    # Hit
                    target['health'] -= p['damage']
                    self._create_particles(target['pos'], 10, p['color'], 0.8, 10)
                    # # Sound: Enemy Hit
                    continue # Projectile is consumed
                
                p['vx'] = dx / dist
                p['vy'] = dy / dist
                p['pos'][0] += p['vx'] * p['speed']
                p['pos'][1] += p['vy'] * p['speed']

            projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep
    
    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                reward += 1.0
                self.score += 10
                self.resources += 2
                self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY, 1.5, 15)
                # # Sound: Enemy Explode
                continue

            path_idx = enemy['path_index']
            if path_idx >= len(self.path_coords):
                self.base_health = max(0, self.base_health - 10)
                self.last_base_hit_time = self.steps
                self._create_particles(self._iso_to_screen(0,3), 40, self.COLOR_BASE_DMG, 2, 20)
                # # Sound: Base Hit
                continue

            target_grid_pos = self.path_coords[path_idx]
            target_screen_pos = self._iso_to_screen(*target_grid_pos)

            dx = target_screen_pos[0] - enemy['pos'][0]
            dy = target_screen_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < 4:
                enemy['path_index'] += 1
            else:
                move_dist = min(dist, enemy['speed'] * self.TILE_WIDTH_HALF)
                enemy['pos'][0] += (dx / dist) * move_dist
                enemy['pos'][1] += (dy / dist) * move_dist
            
            enemies_to_keep.append(enemy)
        
        self.enemies = enemies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= p['decay']
    
    def _create_particles(self, pos, count, color, speed_mult=1.0, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(life // 2, life),
                'radius': self.np_random.uniform(2, 5),
                'decay': 0.1,
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
                is_path = (x, y) in self.path_coords
                is_placement = (x, y) in self.placement_tiles
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                if is_placement:
                    color = tuple(min(255, c + 10) for c in color)
                self._draw_iso_tile(self.screen, (x, y), color)
        
        # Draw base
        base_color = self.COLOR_BASE
        if self.steps - self.last_base_hit_time < 10:
            base_color = self.COLOR_BASE_DMG
        self._draw_iso_cube(self.screen, (0, 3, 0), 1, 1, 0.5, base_color)
        
        # Draw towers
        for tower in self.towers:
            color = self.COLOR_TOWER_1 if tower['type'] == 1 else self.COLOR_TOWER_2
            self._draw_iso_cube(self.screen, (tower['grid_pos'][0], tower['grid_pos'][1], 0), 0.8, 0.8, 0.7, color)

        # Draw enemies (sorted by y-pos for correct layering)
        sorted_enemies = sorted(self.enemies, key=lambda e: e['pos'][1])
        for enemy in sorted_enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 6)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 12
            pygame.draw.rect(self.screen, (50, 50, 50), (pos[0] - bar_width/2, pos[1] - 12, bar_width, 3))
            pygame.draw.rect(self.screen, (50, 200, 50), (pos[0] - bar_width/2, pos[1] - 12, bar_width * health_ratio, 3))

        # Draw cursor
        cursor_tuple = tuple(self.cursor_pos)
        is_valid = cursor_tuple in self.placement_tiles and not any(t['grid_pos'] == cursor_tuple for t in self.towers)
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_ENEMY
        self._draw_iso_tile(self.screen, self.cursor_pos, cursor_color, outline=True)

        # Draw projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), 3)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, p['color'])

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _draw_iso_tile(self, surface, pos, color, outline=False):
        gx, gy = pos
        sx, sy = self._iso_to_screen(gx, gy)
        points = [
            (sx, sy + self.TILE_HEIGHT_HALF),
            (sx + self.TILE_WIDTH_HALF, sy),
            (sx, sy - self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy)
        ]
        if outline:
            pygame.draw.lines(surface, color, True, points, 2)
        else:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_cube(self, surface, pos, w, d, h, color):
        gx, gy, gz = pos
        x, y = self._iso_to_screen(gx, gy)
        y -= gz * self.TILE_HEIGHT_HALF * 2
        
        w_half = self.TILE_WIDTH_HALF * w
        d_half = self.TILE_HEIGHT_HALF * d
        h_px = self.TILE_HEIGHT_HALF * 2 * h

        top_points = [
            (x, y - d_half), (x + w_half, y), (x, y + d_half), (x - w_half, y)
        ]
        
        c_dark = tuple(c * 0.6 for c in color)
        c_med = tuple(c * 0.8 for c in color)
        
        # Left face
        left_points = [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + h_px), (top_points[3][0], top_points[3][1] + h_px)]
        pygame.gfxdraw.filled_polygon(surface, left_points, c_dark)
        # Right face
        right_points = [top_points[2], top_points[1], (top_points[1][0], top_points[1][1] + h_px), (top_points[2][0], top_points[2][1] + h_px)]
        pygame.gfxdraw.filled_polygon(surface, right_points, c_med)
        # Top face
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / 100
        bar_color = (int(200 - 150 * health_ratio), int(50 + 150 * health_ratio), 50)
        pygame.draw.rect(self.screen, (40, 40, 40), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, bar_color, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_small.render(f"BASE HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Resources
        res_text = self.font_small.render(f"RESOURCES: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (220, 12))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 120, 12))

        # Wave Info
        if self.wave_active:
            wave_text = self.font_small.render(f"WAVE {self.current_wave}/5", True, self.COLOR_TEXT)
        elif self.win:
            wave_text = self.font_small.render("ALL WAVES CLEARED", True, self.COLOR_BASE)
        elif self.game_over and not self.win:
             wave_text = self.font_small.render("BASE DESTROYED", True, self.COLOR_ENEMY)
        else:
            wave_text = self.font_small.render(f"NEXT WAVE IN {self.wave_timer / 30:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH / 2 - wave_text.get_width() / 2, 12))

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH / 2 - end_text.get_width() / 2, self.HEIGHT / 2 - end_text.get_height() / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "current_wave": self.current_wave,
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To run the game with manual controls
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        # Place towers
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # 30 FPS

    env.close()