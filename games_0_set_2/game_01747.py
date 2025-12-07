
# Generated: 2025-08-28T02:35:20.728150
# Source Brief: brief_01747.md
# Brief Index: 1747

        
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
        "Controls: ↑↓←→ to move cursor. Space to place tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers in an isometric 2D world."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 12
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.ISO_ORIGIN = (self.SCREEN_WIDTH // 2, 60)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 64)
        self.COLOR_GRID = (35, 40, 52)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)

        # Game constants
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.TOTAL_WAVES = 10
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_RESOURCES = 300
        
        # Define Tower Types
        self.TOWER_TYPES = [
            {
                "name": "Gatling", "cost": 100, "range": 80, "damage": 5,
                "fire_rate": 0.2, "color": (0, 200, 255), "proj_speed": 8, "proj_size": 2
            },
            {
                "name": "Cannon", "cost": 250, "range": 120, "damage": 25,
                "fire_rate": 1.5, "color": (255, 150, 0), "proj_speed": 6, "proj_size": 4
            }
        ]

        # Define Enemy Path (grid coordinates)
        self.enemy_path_grid = [
            (0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (4, 3), (5, 3),
            (6, 3), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (8, 7), (9, 7),
            (10, 7), (11, 7), (12, 7), (12, 6), (12, 5), (12, 4), (13, 4),
            (14, 4), (15, 4), (16, 4), (16, 5), (16, 6), (17, 6), (18, 6), (19, 6)
        ]
        self.enemy_path_screen = [self._iso_to_screen(gx, gy) for gx, gy in self.enemy_path_grid]
        
        self.base_pos_grid = self.enemy_path_grid[-1]
        self.base_pos_screen = self.enemy_path_screen[-1]

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.spawn_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ISO_ORIGIN[0] + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ISO_ORIGIN[1] + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        self.spawn_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0

        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.grid = [[{'buildable': True, 'occupied': False} for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        for gx, gy in self.enemy_path_grid:
            self.grid[gx][gy]['buildable'] = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty

        # 1. Handle player input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self._handle_input(movement, space_pressed, shift_pressed)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # 2. Update game logic
        reward += self._update_waves()
        self._update_towers()
        reward += self._update_enemies()
        reward += self._update_projectiles()
        self._update_particles()
        
        self.score += reward
        
        # 3. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.base_health <= 0:
                reward -= 100
            elif self.current_wave > self.TOTAL_WAVES:
                reward += 100
            self.score += reward
            self.game_over = True
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle tower
        if shift_pressed:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
            # sfx: UI_CYCLE

        # Place tower
        if space_pressed:
            cx, cy = self.cursor_pos
            tower_type = self.TOWER_TYPES[self.selected_tower_idx]
            if self.grid[cx][cy]['buildable'] and not self.grid[cx][cy]['occupied'] and self.resources >= tower_type['cost']:
                self.resources -= tower_type['cost']
                screen_pos = self._iso_to_screen(cx, cy)
                self.towers.append({
                    'type_idx': self.selected_tower_idx,
                    'pos': screen_pos,
                    'cooldown': 0
                })
                self.grid[cx][cy]['occupied'] = True
                # sfx: TOWER_PLACE
                self._create_particles(screen_pos, self.COLOR_CURSOR, 20, 2)

    def _update_waves(self):
        if self.current_wave > self.TOTAL_WAVES:
            return 0
            
        if self.wave_timer > 0:
            self.wave_timer -= 1
            if self.wave_timer == 0:
                self.current_wave += 1
                self.enemies_in_wave = 3 + self.current_wave * 2
                self.enemies_spawned = 0
                self.spawn_timer = 0
                # sfx: WAVE_START
        elif self.enemies_spawned < self.enemies_in_wave:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.spawn_timer = 30 # Time between enemies in a wave
        elif not self.enemies: # Wave cleared
            self.wave_timer = 150 # Time between waves
        return 0

    def _spawn_enemy(self):
        self.enemies_spawned += 1
        difficulty_mod = 1 + (self.current_wave - 1) * 0.05
        self.enemies.append({
            'pos': list(self.enemy_path_screen[0]),
            'path_idx': 0,
            'health': int(50 * difficulty_mod),
            'max_health': int(50 * difficulty_mod),
            'speed': 1.0 * difficulty_mod,
            'reward': 10 + self.current_wave
        })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1 / 30)
            if tower['cooldown'] > 0:
                continue

            tower_type = self.TOWER_TYPES[tower['type_idx']]
            target = None
            # Target enemy furthest along the path
            best_path_idx = -1
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist <= tower_type['range'] and enemy['path_idx'] > best_path_idx:
                    best_path_idx = enemy['path_idx']
                    target = enemy
            
            if target:
                self.projectiles.append({
                    'pos': list(tower['pos']),
                    'target_pos': list(target['pos']),
                    'type_idx': tower['type_idx'],
                })
                tower['cooldown'] = tower_type['fire_rate']
                # sfx: TOWER_FIRE

    def _update_enemies(self):
        reward = 0
        for enemy in reversed(self.enemies):
            if enemy['path_idx'] >= len(self.enemy_path_screen) - 1:
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                reward -= 10
                # sfx: BASE_HIT
                self._create_particles(self.base_pos_screen, self.COLOR_ENEMY, 50, 4)
                continue

            target_pos = self.enemy_path_screen[enemy['path_idx'] + 1]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['path_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in reversed(self.projectiles):
            proj_type = self.TOWER_TYPES[proj['type_idx']]
            dx = proj['target_pos'][0] - proj['pos'][0]
            dy = proj['target_pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj_type['proj_speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'][0] += (dx / dist) * proj_type['proj_speed']
                proj['pos'][1] += (dy / dist) * proj_type['proj_speed']

            # Check for collision with any enemy
            hit = False
            for enemy in reversed(self.enemies):
                e_dist = math.hypot(enemy['pos'][0] - proj['pos'][0], enemy['pos'][1] - proj['pos'][1])
                if e_dist < 10: # Hit radius
                    enemy['health'] -= proj_type['damage']
                    reward += 0.1
                    hit = True
                    # sfx: ENEMY_HIT
                    self._create_particles(proj['pos'], proj_type['color'], 5, 1)
                    if enemy['health'] <= 0:
                        reward += 1
                        self.resources += enemy['reward']
                        self.enemies.remove(enemy)
                        # sfx: ENEMY_DESTROY
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 30, 3)
                    break # Projectile can only hit one enemy
            
            if hit or dist < proj_type['proj_speed']:
                 self.projectiles.remove(proj)
        return reward

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.current_wave > self.TOTAL_WAVES

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_path = not self.grid[x][y]['buildable']
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), color)

        # 2. Collect and sort all drawable entities by screen y-pos for correct layering
        render_list = []
        for tower in self.towers:
            render_list.append({'type': 'tower', 'y': tower['pos'][1], 'data': tower})
        for enemy in self.enemies:
            render_list.append({'type': 'enemy', 'y': enemy['pos'][1], 'data': enemy})
        
        # Add base to render list
        base_y = self.base_pos_screen[1]
        render_list.append({'type': 'base', 'y': base_y, 'data': None})
        
        render_list.sort(key=lambda item: item['y'])

        # 3. Draw entities in sorted order
        for item in render_list:
            if item['type'] == 'tower':
                self._render_tower(item['data'])
            elif item['type'] == 'enemy':
                self._render_enemy(item['data'])
            elif item['type'] == 'base':
                self._render_base()
        
        # 4. Draw projectiles and particles on top
        for proj in self.projectiles:
            proj_type = self.TOWER_TYPES[proj['type_idx']]
            pygame.draw.circle(self.screen, proj_type['color'], (int(proj['pos'][0]), int(proj['pos'][1])), proj_type['proj_size'])

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size'] * (p['life'] / 20.0)))
            
        # 5. Draw cursor
        cx, cy = self.cursor_pos
        tower_type = self.TOWER_TYPES[self.selected_tower_idx]
        can_build = self.grid[cx][cy]['buildable'] and not self.grid[cx][cy]['occupied'] and self.resources >= tower_type['cost']
        cursor_color = self.COLOR_CURSOR if can_build else self.COLOR_CURSOR_INVALID
        
        p1 = self._iso_to_screen(cx, cy)
        p2 = self._iso_to_screen(cx + 1, cy)
        p3 = self._iso_to_screen(cx + 1, cy + 1)
        p4 = self._iso_to_screen(cx, cy + 1)
        pygame.draw.polygon(self.screen, cursor_color, (p1, p2, p3, p4), 2)

    def _render_base(self):
        x, y = self.base_pos_screen
        pygame.draw.circle(self.screen, self.COLOR_BASE, (x, y), 12)
        pygame.gfxdraw.aacircle(self.screen, x, y, 12, self.COLOR_BASE)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 4)

    def _render_tower(self, tower):
        tower_type = self.TOWER_TYPES[tower['type_idx']]
        x, y = tower['pos']
        base_color = (int(c*0.6) for c in tower_type['color'])
        pygame.draw.circle(self.screen, tuple(base_color), (x, y), 8)
        pygame.draw.circle(self.screen, tower_type['color'], (x, y), 5)
        
    def _render_enemy(self, enemy):
        x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
        pygame.draw.circle(self.screen, self.COLOR_ENEMY, (x, y), 6)
        pygame.gfxdraw.aacircle(self.screen, x, y, 6, self.COLOR_ENEMY)

        # Health bar
        health_ratio = enemy['health'] / enemy['max_health']
        bar_w = 12
        bar_h = 3
        bar_x = x - bar_w // 2
        bar_y = y - 12
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_w * health_ratio), bar_h))

    def _render_ui(self):
        # Base Health
        health_text = self.font_small.render(f"Base HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Resources
        res_text = self.font_small.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 30))

        # Wave Info
        if self.current_wave > self.TOTAL_WAVES:
            wave_str = "VICTORY!"
        elif self.wave_timer > 0 and self.current_wave < self.TOTAL_WAVES:
            wave_str = f"Wave {self.current_wave + 1} in {self.wave_timer/30:.1f}s"
        else:
            wave_str = f"Wave {self.current_wave}/{self.TOTAL_WAVES}"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Selected Tower Info
        tower = self.TOWER_TYPES[self.selected_tower_idx]
        name_text = self.font_large.render(tower['name'], True, self.COLOR_TEXT)
        cost_text = self.font_small.render(f"Cost: ${tower['cost']}", True, self.COLOR_TEXT)
        stats_text = self.font_small.render(f"DMG: {tower['damage']} | Range: {tower['range']} | Rate: {tower['fire_rate']:.1f}s", True, self.COLOR_TEXT)
        
        self.screen.blit(name_text, (10, self.SCREEN_HEIGHT - 60))
        self.screen.blit(cost_text, (10, self.SCREEN_HEIGHT - 35))
        self.screen.blit(stats_text, (10, self.SCREEN_HEIGHT - 20))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_str = "VICTORY" if self.base_health > 0 else "GAME OVER"
            result_text = self.font_large.render(result_str, True, self.COLOR_CURSOR if self.base_health > 0 else self.COLOR_ENEMY)
            text_rect = result_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # Use arrow keys for movement, space to place, left shift to cycle
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    env.close()