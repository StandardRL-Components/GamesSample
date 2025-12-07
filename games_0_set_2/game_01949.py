
# Generated: 2025-08-28T03:11:35.895552
# Source Brief: brief_01949.md
# Brief Index: 1949

        
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
        "Controls: Arrow keys to move the placement cursor. Press space to build a tower. "
        "The tower type alternates with each build."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing "
        "damage and slowing towers along their path. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 70)
    COLOR_BASE = (50, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_GRID_VALID = (0, 255, 0, 50)
    
    COLOR_ENEMY_BODY = (220, 50, 50)
    COLOR_ENEMY_HEALTH = (255, 60, 60)
    COLOR_ENEMY_SLOWED = (100, 150, 255)

    TOWER_SLOW_COLOR = (60, 120, 255)
    TOWER_DMG_COLOR = (255, 200, 50)
    
    UI_TEXT_COLOR = (220, 220, 240)
    UI_HEALTH_BAR_COLOR = (50, 200, 100)
    UI_HEALTH_BAR_BG_COLOR = (100, 40, 40)
    
    # Game settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 12000 # Increased for 10 waves at 30fps
    TOTAL_WAVES = 10
    BASE_MAX_HEALTH = 100
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Game path and grid
        self._define_path_and_grid()
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.base_damage_flash = 0
        self.wave_number = 0
        self.wave_cooldown = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.next_tower_type = "damage"
        self.prev_space_held = False
        self.reward_this_step = 0
        
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def _define_path_and_grid(self):
        self.grid_size = 40
        self.cols = self.SCREEN_WIDTH // self.grid_size
        self.rows = self.SCREEN_HEIGHT // self.grid_size

        path_coords = [
            (-1, 5), (1, 5), (1, 2), (4, 2), (4, 8), (7, 8), (7, 1),
            (11, 1), (11, 6), (13, 6), (13, 3), (16, 3)
        ]
        self.path = [(c * self.grid_size + self.grid_size // 2, r * self.grid_size + self.grid_size // 2) for c, r in path_coords]
        
        path_grid_coords = set()
        for i in range(len(path_coords) - 1):
            x1, y1 = path_coords[i]
            x2, y2 = path_coords[i+1]
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path_grid_coords.add((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path_grid_coords.add((x2, y))

        self.placement_spots = []
        for r in range(self.rows):
            for c in range(self.cols):
                is_path = (c,r) in path_grid_coords or (c-1,r) in path_grid_coords or (c+1,r) in path_grid_coords or (c,r-1) in path_grid_coords or (c,r+1) in path_grid_coords
                is_ui_area = r < 1
                if not is_path and not is_ui_area:
                    self.placement_spots.append((c, r))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.BASE_MAX_HEALTH
        self.base_damage_flash = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.cols // 2, self.rows // 2]
        self.next_tower_type = "damage"
        self.prev_space_held = True # Prevent building on first frame
        self.reward_this_step = 0

        self.wave_number = 0
        self.wave_cooldown = 90 # 3 seconds at 30fps
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if not self.game_over:
            # --- Handle Input ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held)
            
            # --- Update Game Logic ---
            self._update_waves()
            self._update_enemies()
            self._update_towers()
            self._update_projectiles()
            
            # Continuous survival reward
            self.reward_this_step -= 0.001

        # --- Visuals and State Update ---
        self._update_particles()
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1
            
        self.steps += 1
        
        # --- Termination and Rewards ---
        terminated = self._check_termination()
        reward = self.reward_this_step
        
        if terminated:
            if self.base_health <= 0:
                reward = -100.0
            elif self.wave_number > self.TOTAL_WAVES:
                reward = 100.0
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.cols - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.rows - 1)

        # --- Tower Placement ---
        if space_held and not self.prev_space_held:
            cursor_grid_pos = tuple(self.cursor_pos)
            is_valid_spot = cursor_grid_pos in self.placement_spots
            is_occupied = any(t['grid_pos'] == cursor_grid_pos for t in self.towers)
            
            if is_valid_spot and not is_occupied:
                self._place_tower(cursor_grid_pos)

        self.prev_space_held = space_held

    def _place_tower(self, grid_pos):
        pos = (grid_pos[0] * self.grid_size + self.grid_size // 2, grid_pos[1] * self.grid_size + self.grid_size // 2)
        if self.next_tower_type == "damage":
            tower = {'type': 'damage', 'pos': pos, 'grid_pos': grid_pos, 'range': 100, 'cooldown': 0, 'max_cooldown': 20}
            self.next_tower_type = "slow"
        else: # slow
            tower = {'type': 'slow', 'pos': pos, 'grid_pos': grid_pos, 'range': 80, 'cooldown': 0, 'max_cooldown': 45}
            self.next_tower_type = "damage"
        self.towers.append(tower)
        
        # Sound placeholder: # sfx_build_tower()
        self._create_particles(pos, 15, self.TOWER_DMG_COLOR if tower['type'] == 'damage' else self.TOWER_SLOW_COLOR, 1, 3)

    def _update_waves(self):
        if not self.enemies and self.wave_number <= self.TOTAL_WAVES and not self.game_over:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self._spawn_wave()
                    self.wave_cooldown = 150 # 5s between waves

    def _spawn_wave(self):
        num_enemies = 5 + self.wave_number * 2
        health = 1 + self.wave_number
        speed = 0.5 + self.wave_number * 0.05
        for i in range(num_enemies):
            self.enemies.append({
                'pos': list(self.path[0]),
                'path_index': 0,
                'health': health,
                'max_health': health,
                'speed': speed,
                'base_speed': speed,
                'slow_timer': 0,
                'spawn_delay': i * 15, # Stagger spawn
                'size': 10
            })
        # Sound placeholder: # sfx_wave_start()

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['spawn_delay'] > 0:
                enemy['spawn_delay'] -= 1
                continue
            
            if enemy['slow_timer'] > 0:
                enemy['slow_timer'] -= 1
                enemy['speed'] = enemy['base_speed'] * 0.5
            else:
                enemy['speed'] = enemy['base_speed']

            if enemy['path_index'] < len(self.path) - 1:
                target = self.path[enemy['path_index'] + 1]
                dx = target[0] - enemy['pos'][0]
                dy = target[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy['speed']:
                    enemy['path_index'] += 1
                else:
                    enemy['pos'][0] += (dx / dist) * enemy['speed']
                    enemy['pos'][1] += (dy / dist) * enemy['speed']
            else:
                # Reached base
                self.base_health -= enemy['health']
                self.base_damage_flash = 10
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], 20, self.COLOR_ENEMY_BODY, 2, 4)
                # Sound placeholder: # sfx_base_damage()

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            # Find target
            target = None
            min_dist = tower['range']
            for enemy in self.enemies:
                if enemy['spawn_delay'] > 0: continue
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['max_cooldown']
                if tower['type'] == 'damage':
                    self.projectiles.append({
                        'start_pos': list(tower['pos']),
                        'pos': list(tower['pos']),
                        'target': target,
                        'speed': 8,
                        'damage': 1
                    })
                    # Sound placeholder: # sfx_shoot_damage()
                elif tower['type'] == 'slow':
                    target['slow_timer'] = 90 # 3 seconds
                    self._create_particles(target['pos'], 10, self.TOWER_SLOW_COLOR, 0.5, 2, life=15)
                    # Sound placeholder: # sfx_shoot_slow()

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue
            
            dx = p['target']['pos'][0] - p['pos'][0]
            dy = p['target']['pos'][1] - p['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                self.reward_this_step += 0.1
                self._create_particles(p['target']['pos'], 5, (255, 255, 255), 1, 2, life=10)
                # Sound placeholder: # sfx_enemy_hit()
                if p['target']['health'] <= 0:
                    self.reward_this_step += 1.0
                    self._create_particles(p['target']['pos'], 30, self.COLOR_ENEMY_HEALTH, 1, 4)
                    self.enemies.remove(p['target'])
                    # Sound placeholder: # sfx_enemy_destroy()
                self.projectiles.remove(p)
            else:
                p['pos'][0] += (dx / dist) * p['speed']
                p['pos'][1] += (dy / dist) * p['speed']

    def _create_particles(self, pos, count, color, min_speed, max_speed, life=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.TOTAL_WAVES and not self.enemies:
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
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, width=self.grid_size)
        
        # Base
        base_color = self.COLOR_BASE_DMG if self.base_damage_flash > 0 else self.COLOR_BASE
        base_rect = pygame.Rect(self.path[-1][0] - self.grid_size // 2, self.path[-1][1] - self.grid_size // 2, self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, base_color, base_rect)
        
        # Placement spots
        for c, r in self.placement_spots:
            center = (c * self.grid_size + self.grid_size // 2, r * self.grid_size + self.grid_size // 2)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 3, self.COLOR_GRID_VALID)

        # Towers
        for tower in self.towers:
            color = self.TOWER_DMG_COLOR if tower['type'] == 'damage' else self.TOWER_SLOW_COLOR
            if tower['type'] == 'damage':
                points = [
                    (tower['pos'][0], tower['pos'][1] - 10),
                    (tower['pos'][0] - 8, tower['pos'][1] + 6),
                    (tower['pos'][0] + 8, tower['pos'][1] + 6),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            else:
                pygame.gfxdraw.aacircle(self.screen, int(tower['pos'][0]), int(tower['pos'][1]), 8, color)
                pygame.gfxdraw.filled_circle(self.screen, int(tower['pos'][0]), int(tower['pos'][1]), 8, color)

        # Enemies
        for enemy in self.enemies:
            if enemy['spawn_delay'] > 0: continue
            rect = pygame.Rect(0, 0, enemy['size'], enemy['size'])
            rect.center = enemy['pos']
            color = self.COLOR_ENEMY_SLOWED if enemy['slow_timer'] > 0 else self.COLOR_ENEMY_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            # Health bar
            if enemy['health'] < enemy['max_health']:
                health_pct = enemy['health'] / enemy['max_health']
                bar_w = int(enemy['size'] * 1.5)
                bar_h = 3
                bar_x = enemy['pos'][0] - bar_w // 2
                bar_y = enemy['pos'][1] - enemy['size'] - 3
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (bar_x, bar_y, bar_w * health_pct, bar_h))

        # Projectiles
        for p in self.projectiles:
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (255, 255, 255))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (255, 255, 255))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = max(1, int(3 * (p['life'] / p['max_life'])))
            pygame.draw.circle(self.screen, color, p['pos'], size)
            
        # Cursor
        cursor_center = (self.cursor_pos[0] * self.grid_size + self.grid_size // 2, self.cursor_pos[1] * self.grid_size + self.grid_size // 2)
        cursor_color = self.TOWER_DMG_COLOR if self.next_tower_type == 'damage' else self.TOWER_SLOW_COLOR
        pygame.draw.rect(self.screen, cursor_color, (cursor_center[0]-12, cursor_center[1]-12, 24, 24), 2, border_radius=4)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_str = f"WAVE: {min(self.wave_number, self.TOTAL_WAVES)}/{self.TOTAL_WAVES}"
        if self.wave_cooldown > 0 and self.wave_number <= self.TOTAL_WAVES and not self.enemies:
            wave_str = f"WAVE {self.wave_number} IN {self.wave_cooldown / 30:.1f}s"
        wave_text = self.font_small.render(wave_str, True, self.UI_TEXT_COLOR)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Base Health
        health_pct = max(0, self.base_health / self.BASE_MAX_HEALTH)
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH // 2 - bar_width // 2
        bar_y = 10
        pygame.draw.rect(self.screen, self.UI_HEALTH_BAR_BG_COLOR, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.UI_HEALTH_BAR_COLOR, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=4)
        health_text = self.font_small.render(f"{int(self.base_health)}/{self.BASE_MAX_HEALTH}", True, self.UI_TEXT_COLOR)
        self.screen.blit(health_text, (bar_x + bar_width // 2 - health_text.get_width() // 2, bar_y))

        # Game Over / Win Message
        if self.game_over:
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY_HEALTH
            else:
                msg = "YOU WIN!"
                color = self.COLOR_BASE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # action is a list [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Event Handling for Human Play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key
        
        # Get key presses for this frame
        keys = pygame.key.get_pressed()
        
        # Reset movement action
        action[0] = 0
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Wait for a moment on the end screen before auto-resetting
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            obs, info = env.reset()

        # --- Rendering for Human Play ---
        # The observation is the rendered frame, so we just display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the framerate
        clock.tick(30)
        
    env.close()