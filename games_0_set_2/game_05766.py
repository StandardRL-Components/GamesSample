
# Generated: 2025-08-28T06:01:29.197834
# Source Brief: brief_05766.md
# Brief Index: 5766

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press space to build the selected tower. Hold shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric tower defense game. Place towers to defend your base "
        "from waves of incoming enemies. Manage your gold and survive all 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = 80
    MAX_STEPS = 3000 # Increased from 1000 to allow for longer games

    # --- Colors ---
    COLOR_BG = (25, 35, 50)
    COLOR_GRID = (40, 55, 70)
    COLOR_PATH = (60, 80, 100)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 100)
    COLOR_TEXT = (230, 240, 255)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_CURSOR_VALID = (50, 255, 50)
    COLOR_CURSOR_INVALID = (255, 50, 50)
    
    # --- Tower and Enemy Data ---
    TOWER_TYPES = [
        {'name': 'Gatling', 'cost': 50, 'range': 100, 'damage': 5, 'fire_rate': 8, 'color': (0, 200, 200), 'proj_speed': 8},
        {'name': 'Cannon', 'cost': 120, 'range': 150, 'damage': 25, 'fire_rate': 40, 'color': (255, 150, 0), 'proj_speed': 6},
    ]
    PATH_COORDS = [(0, 5), (3, 5), (3, 2), (7, 2), (7, 8), (12, 8), (12, 4), (17, 4), (17, 6), (19, 6)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.pixel_path = [self._iso_to_screen(x, y) for x, y in self.PATH_COORDS]
        
        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        iso_x = (grid_x - grid_y) * self.TILE_WIDTH_HALF
        iso_y = (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(iso_x + self.ISO_OFFSET_X), int(iso_y + self.ISO_OFFSET_Y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.gold = 150 # Starting gold
        self.wave_number = 0
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for x, y in self.PATH_COORDS:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x][y] = -1 # Path is non-buildable

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.wave_in_progress = False
        self.wave_cooldown = 90 # Frames before first wave
        self.enemies_to_spawn = 0
        self.spawn_cooldown = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > 10:
            self.win = True
            return

        num_enemies = 4 + self.wave_number
        self.enemy_health = 20 * (1.05 ** (self.wave_number - 1))
        self.enemy_speed = 0.8 * (1.02 ** (self.wave_number - 1))
        
        self.enemies_to_spawn = num_enemies
        self.spawn_cooldown = 0
        self.wave_in_progress = True

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        step_reward = 0.0

        self._handle_input(movement, space_held, shift_held)
        
        if not self.game_over:
            # --- Game Logic Updates ---
            self._update_spawner()
            step_reward += self._update_enemies()
            step_reward += self._update_projectiles()
            self._update_towers()
            self._update_particles()
            
            # --- Wave Management ---
            if self.wave_in_progress and not self.enemies and self.enemies_to_spawn == 0:
                self.wave_in_progress = False
                self.wave_cooldown = 180 # Time until next wave
                if self.wave_number <= 10:
                    step_reward += 1.0 # Wave completion reward

            if not self.wave_in_progress and not self.win:
                self.wave_cooldown -= 1
                if self.wave_cooldown <= 0:
                    self._start_next_wave()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.win:
                step_reward += 50.0
            elif self.base_health <= 0:
                step_reward += -50.0
        
        self.score += step_reward
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Place Tower ---
        if space_held and not self.last_space_held:
            self._place_tower()
        self.last_space_held = space_held

        # --- Cycle Tower Type ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.last_shift_held = shift_held
    
    def _is_placement_valid(self):
        cx, cy = self.cursor_pos
        selected_tower = self.TOWER_TYPES[self.selected_tower_type]
        return self.grid[cx][cy] == 0 and self.gold >= selected_tower['cost']

    def _place_tower(self):
        if self._is_placement_valid():
            cx, cy = self.cursor_pos
            tower_data = self.TOWER_TYPES[self.selected_tower_type]
            
            self.gold -= tower_data['cost']
            self.grid[cx][cy] = self.selected_tower_type + 1 # Mark grid as occupied
            
            new_tower = {
                'grid_pos': [cx, cy],
                'pixel_pos': self._iso_to_screen(cx, cy),
                'type': self.selected_tower_type,
                'cooldown': 0,
            }
            self.towers.append(new_tower)
            # sfx: tower_place.wav
            self._create_particle_burst(new_tower['pixel_pos'], 20, tower_data['color'], 2, 15)

    def _update_spawner(self):
        if self.enemies_to_spawn > 0 and self.wave_in_progress:
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0:
                start_pos = list(self.pixel_path[0])
                self.enemies.append({
                    'pos': start_pos,
                    'health': self.enemy_health,
                    'max_health': self.enemy_health,
                    'speed': self.enemy_speed,
                    'path_index': 0,
                    'dist_on_segment': 0.0,
                    'value': 10 + self.wave_number, # Gold value
                })
                self.enemies_to_spawn -= 1
                self.spawn_cooldown = 30 # Frames between spawns

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            path_idx = enemy['path_index']
            if path_idx >= len(self.pixel_path) - 1:
                self.enemies.remove(enemy)
                self.base_health = max(0, self.base_health - 10)
                # sfx: base_damage.wav
                self._create_particle_burst(self.pixel_path[-1], 30, self.COLOR_ENEMY, 3, 20)
                continue

            start_node = self.pixel_path[path_idx]
            end_node = self.pixel_path[path_idx + 1]
            
            segment_vec = (end_node[0] - start_node[0], end_node[1] - start_node[1])
            segment_len = math.hypot(*segment_vec)
            
            enemy['dist_on_segment'] += enemy['speed']
            
            if enemy['dist_on_segment'] >= segment_len:
                enemy['path_index'] += 1
                enemy['dist_on_segment'] = 0
                if enemy['path_index'] < len(self.pixel_path):
                     enemy['pos'] = list(self.pixel_path[enemy['path_index']])
            else:
                progress = enemy['dist_on_segment'] / segment_len if segment_len > 0 else 0
                enemy['pos'][0] = start_node[0] + segment_vec[0] * progress
                enemy['pos'][1] = start_node[1] + segment_vec[1] * progress
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                tower_data = self.TOWER_TYPES[tower['type']]
                target = None
                best_dist = float('inf')

                for enemy in self.enemies:
                    dist = math.hypot(tower['pixel_pos'][0] - enemy['pos'][0], tower['pixel_pos'][1] - enemy['pos'][1])
                    if dist <= tower_data['range']:
                        # Prioritize enemy furthest along the path
                        enemy_path_dist = enemy['path_index'] + (enemy['dist_on_segment'] / 1000.0)
                        if enemy_path_dist < best_dist:
                            best_dist = enemy_path_dist
                            target = enemy
                
                if target:
                    # sfx: shoot.wav
                    tower['cooldown'] = tower_data['fire_rate']
                    self.projectiles.append({
                        'pos': list(tower['pixel_pos']),
                        'target': target,
                        'type': tower['type']
                    })

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            tower_data = self.TOWER_TYPES[proj['type']]
            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            dx = target_pos[0] - proj_pos[0]
            dy = target_pos[1] - proj_pos[1]
            dist = math.hypot(dx, dy)
            
            if dist < tower_data['proj_speed']:
                proj['target']['health'] -= tower_data['damage']
                self._create_particle_burst(proj['pos'], 5, tower_data['color'], 1.5, 10)
                # sfx: hit.wav
                if proj['target']['health'] <= 0:
                    self.gold += proj['target']['value']
                    self._create_particle_burst(proj['target']['pos'], 25, self.COLOR_ENEMY, 2, 25)
                    self.enemies.remove(proj['target'])
                    reward += 0.1 # Kill reward
                    # sfx: enemy_die.wav
                self.projectiles.remove(proj)
            else:
                proj_pos[0] += (dx / dist) * tower_data['proj_speed']
                proj_pos[1] += (dy / dist) * tower_data['proj_speed']
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "gold": self.gold, "wave": self.wave_number }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_path()
        self._render_base()
        
        # Render entities, sorted by y-coordinate for correct isometric overlap
        render_queue = []
        for t in self.towers:
            render_queue.append({'y': t['pixel_pos'][1], 'type': 'tower', 'data': t})
        for e in self.enemies:
            render_queue.append({'y': e['pos'][1], 'type': 'enemy', 'data': e})
        
        render_queue.sort(key=lambda item: item['y'])

        for item in render_queue:
            if item['type'] == 'tower':
                self._render_tower(item['data'])
            elif item['type'] == 'enemy':
                self._render_enemy(item['data'])

        self._render_cursor()
        for proj in self.projectiles: self._render_projectile(proj)
        for part in self.particles: self._render_particle(part)

    def _render_grid_and_path(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                px, py = self._iso_to_screen(x, y)
                points = [
                    (px, py - self.TILE_HEIGHT_HALF),
                    (px + self.TILE_WIDTH_HALF, py),
                    (px, py + self.TILE_HEIGHT_HALF),
                    (px - self.TILE_WIDTH_HALF, py)
                ]
                color = self.COLOR_PATH if self.grid[x][y] == -1 else self.COLOR_GRID
                pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_base(self):
        px, py = self.pixel_path[-1]
        pygame.gfxdraw.filled_circle(self.screen, px, py, 15, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, px, py, 15, self.COLOR_BASE)
        
        # Pulsing glow effect
        glow_radius = 18 + int(math.sin(self.steps * 0.1) * 3)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, self.COLOR_BASE_GLOW)
        self.screen.blit(temp_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_tower(self, tower):
        px, py = tower['pixel_pos']
        tower_data = self.TOWER_TYPES[tower['type']]
        color = tower_data['color']
        
        if tower_data['name'] == 'Gatling':
            pygame.gfxdraw.filled_circle(self.screen, px, py - 5, 6, color)
            pygame.gfxdraw.aacircle(self.screen, px, py - 5, 6, color)
            pygame.draw.rect(self.screen, color, (px - 2, py - 10, 4, 10))
        elif tower_data['name'] == 'Cannon':
            pygame.gfxdraw.filled_circle(self.screen, px, py - 8, 8, color)
            pygame.gfxdraw.aacircle(self.screen, px, py - 8, 8, color)
            pygame.draw.rect(self.screen, color, (px - 4, py - 16, 8, 12))

    def _render_enemy(self, enemy):
        px, py = int(enemy['pos'][0]), int(enemy['pos'][1])
        
        # Glow
        glow_radius = 10
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, self.COLOR_ENEMY_GLOW)
        self.screen.blit(temp_surf, (px - glow_radius, py - glow_radius - 5), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Body
        pygame.gfxdraw.filled_trigon(self.screen, px, py-10, px-6, py, px+6, py, self.COLOR_ENEMY)
        pygame.gfxdraw.aatrigon(self.screen, px, py-10, px-6, py, px+6, py, self.COLOR_ENEMY)
        
        # Health bar
        bar_width = 16
        bar_height = 3
        health_ratio = enemy['health'] / enemy['max_health']
        health_width = int(bar_width * health_ratio)
        bar_x, bar_y = px - bar_width // 2, py - 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, health_width, bar_height))

    def _render_projectile(self, proj):
        px, py = int(proj['pos'][0]), int(proj['pos'][1])
        color = self.TOWER_TYPES[proj['type']]['color']
        pygame.gfxdraw.filled_circle(self.screen, px, py, 3, color)
        pygame.gfxdraw.aacircle(self.screen, px, py, 3, color)
    
    def _render_particle(self, particle):
        px, py = int(particle['pos'][0]), int(particle['pos'][1])
        alpha = int(255 * (particle['life'] / particle['max_life']))
        color = (*particle['color'], alpha)
        radius = int(particle['size'] * (particle['life'] / particle['max_life']))
        if radius > 0:
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (px - radius, py - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self._iso_to_screen(cx, cy)
        points = [
            (px, py - self.TILE_HEIGHT_HALF),
            (px + self.TILE_WIDTH_HALF, py),
            (px, py + self.TILE_HEIGHT_HALF),
            (px - self.TILE_WIDTH_HALF, py)
        ]
        color = self.COLOR_CURSOR_VALID if self._is_placement_valid() else self.COLOR_CURSOR_INVALID
        pygame.draw.polygon(self.screen, color, points, 2)
        
        tower_data = self.TOWER_TYPES[self.selected_tower_type]
        range_radius = tower_data['range']
        temp_surf = pygame.Surface((range_radius * 2, range_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(temp_surf, range_radius, range_radius, range_radius, (*color, 60))
        self.screen.blit(temp_surf, (px - range_radius, py - range_radius))


    def _render_ui(self):
        # --- Base Health ---
        hp_text = self.font_large.render(f"Base HP: {int(self.base_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (10, 10))

        # --- Gold ---
        gold_text = self.font_large.render(f"Gold: {self.gold}", True, (255, 223, 0))
        self.screen.blit(gold_text, (self.SCREEN_WIDTH // 2 - gold_text.get_width() // 2, self.SCREEN_HEIGHT - 35))

        # --- Wave Info ---
        wave_str = f"Wave: {min(self.wave_number, 10)}/10"
        if not self.wave_in_progress and not self.win:
            wave_str += f" (Next in {self.wave_cooldown//30}s)"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # --- Selected Tower ---
        tower = self.TOWER_TYPES[self.selected_tower_type]
        select_text = self.font_small.render(f"Build: {tower['name']} (Cost: {tower['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(select_text, (self.SCREEN_WIDTH // 2 - select_text.get_width() // 2, self.SCREEN_HEIGHT - 60))

        # --- Game Over/Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (50, 255, 50) if self.win else (255, 50, 50)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _create_particle_burst(self, pos, count, color, max_speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(life // 2, life),
                'max_life': life,
                'color': color,
                'size': random.uniform(1, 4)
            })

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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
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

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # --- Display the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()