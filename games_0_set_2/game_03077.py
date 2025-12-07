
# Generated: 2025-08-28T06:54:42.967633
# Source Brief: brief_03077.md
# Brief Index: 3077

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through tower types. "
        "Press Space to build the selected tower on the cursor's location or to start the next wave."
    )

    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies. "
        "Manage your resources and survive all 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Screen and Grid Dimensions ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GAME_WIDTH = 480
        self.UI_WIDTH = self.SCREEN_WIDTH - self.GAME_WIDTH
        self.CELL_SIZE = 40
        self.GRID_W = self.GAME_WIDTH // self.CELL_SIZE
        self.GRID_H = self.SCREEN_HEIGHT // self.CELL_SIZE

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PATH = (45, 45, 70)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_UI_BG = (25, 25, 40)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_HEALTH_GREEN = (0, 255, 0)
        self.COLOR_HEALTH_RED = (255, 0, 0)

        # --- Game Constants ---
        self.MAX_WAVES = 10
        self.MAX_STEPS = 15000 # Generous step limit for 10 waves
        self.STARTING_RESOURCES = 100
        self.STARTING_BASE_HEALTH = 20
        self.RESOURCES_PER_WAVE = 50
        
        self._define_path()
        self._define_tower_stats()

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rng = None
        self.game_phase = "INTERMISSION"
        self.wave_number = 0
        self.base_health = 0
        self.resources = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_spawn_timer = 0
        self.wave_enemies_to_spawn = 0
        self.wave_enemies_spawned = 0
        self.base_hit_timer = 0

        self.reset()
        self.validate_implementation()

    def _define_path(self):
        self.path_grid_coords = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (2, 4), (2, 3), (2, 2),
            (3, 2), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5),
            (6, 6), (6, 7), (5, 7), (4, 7), (4, 8), (4, 9), (5, 9),
            (6, 9), (7, 9), (8, 9), (8, 8), (8, 7), (8, 6), (8, 5),
            (8, 4), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3)
        ]
        self.path_pixel_coords = [(c[0] * self.CELL_SIZE + self.CELL_SIZE / 2, c[1] * self.CELL_SIZE + self.CELL_SIZE / 2) for c in self.path_grid_coords]
        self.base_pos_grid = self.path_grid_coords[-1]

    def _define_tower_stats(self):
        self.tower_stats = [
            {'name': 'Gatling', 'cost': 25, 'damage': 1, 'range': 80, 'fire_rate': 0.2, 'color': (0, 150, 255), 'proj_speed': 8},
            {'name': 'Cannon', 'cost': 60, 'damage': 5, 'range': 120, 'fire_rate': 1.5, 'color': (255, 150, 0), 'proj_speed': 6},
            {'name': 'Sniper', 'cost': 80, 'damage': 10, 'range': 200, 'fire_rate': 2.5, 'color': (200, 0, 255), 'proj_speed': 15},
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.game_phase = "INTERMISSION"
        self.wave_number = 0
        self.base_health = self.STARTING_BASE_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = deque(maxlen=200) # Performance optimization
        
        self.grid = [[True for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        for x, y in self.path_grid_coords:
            if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                self.grid[x][y] = False # Path is not buildable
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.base_hit_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.game_over = self._check_termination()
        if self.game_over:
            if self.win:
                reward += 100
            else:
                reward -= 100
            return self._get_observation(), reward, True, False, self._get_info()

        self._handle_input(action)

        if self.game_phase == "WAVE_ACTIVE":
            reward += self._update_wave()
        
        self._update_particles()

        if self.base_hit_timer > 0:
            self.base_hit_timer -= 1

        self.steps += 1
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Actions on Press (rising edge) ---
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        if shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_stats)
            # sfx: UI_cycle.wav

        if space_press:
            if self.game_phase == "INTERMISSION":
                # Check if cursor is on the "Start Wave" button
                start_button_rect = pygame.Rect(self.GAME_WIDTH + 20, self.SCREEN_HEIGHT - 60, self.UI_WIDTH - 40, 40)
                cursor_pixel_pos = (self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE/2)
                
                # A bit of a hack: if cursor is in the rightmost column, it can activate the button
                if self.cursor_pos[0] == self.GRID_W - 1 and 7 <= self.cursor_pos[1] <= 9: 
                    self._start_wave()
                else:
                    self._place_tower()
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        gx, gy = self.cursor_pos
        if self.grid[gx][gy]: # Is buildable
            stats = self.tower_stats[self.selected_tower_type]
            if self.resources >= stats['cost']:
                self.resources -= stats['cost']
                self.grid[gx][gy] = False
                self.towers.append({
                    'gpos': (gx, gy),
                    'px_pos': (gx * self.CELL_SIZE + self.CELL_SIZE / 2, gy * self.CELL_SIZE + self.CELL_SIZE / 2),
                    'type': self.selected_tower_type,
                    'cooldown': 0,
                    'fire_anim_timer': 0
                })
                # sfx: build_tower.wav
    
    def _start_wave(self):
        if self.game_phase == "INTERMISSION":
            self.game_phase = "WAVE_ACTIVE"
            self.wave_number += 1
            
            # Difficulty scaling
            num_enemies = 3 + self.wave_number * 2
            base_health = 5 + self.wave_number * 2
            health_mult = 1.0 + (self.wave_number - 1) * 0.15
            speed_mult = 1.0 + (self.wave_number - 1) * 0.07

            self.wave_enemies_to_spawn = []
            for i in range(num_enemies):
                self.wave_enemies_to_spawn.append({
                    'health': base_health * health_mult,
                    'speed': 1.0 * speed_mult,
                })
            
            self.wave_enemies_spawned = 0
            self.wave_spawn_timer = 0
            # sfx: wave_start.wav

    def _update_wave(self):
        reward = 0
        
        # 1. Spawn Enemies
        self.wave_spawn_timer -= 1 / 30.0 # Assuming 30 FPS
        if self.wave_spawn_timer <= 0 and self.wave_enemies_to_spawn:
            enemy_data = self.wave_enemies_to_spawn.pop(0)
            self.enemies.append({
                'pos': list(self.path_pixel_coords[0]),
                'max_health': enemy_data['health'],
                'health': enemy_data['health'],
                'speed': enemy_data['speed'],
                'path_index': 0,
            })
            self.wave_spawn_timer = 0.5 # Time between spawns
            self.wave_enemies_spawned += 1

        # 2. Update Towers
        self._update_towers()
        
        # 3. Update Projectiles (and get kill rewards)
        reward += self._update_projectiles()

        # 4. Update Enemies
        self._update_enemies()
        
        # 5. Check for Wave End
        if not self.enemies and not self.wave_enemies_to_spawn:
            if self.wave_number >= self.MAX_WAVES:
                self.win = True
            else:
                wave_clear_bonus = 1.0 if self.base_health == self.STARTING_BASE_HEALTH else 0
                reward += wave_clear_bonus
                self.score += wave_clear_bonus
                self.resources += self.RESOURCES_PER_WAVE
                self.game_phase = "INTERMISSION"
                # sfx: wave_clear.wav
        
        return reward

    def _update_towers(self):
        for tower in self.towers:
            stats = self.tower_stats[tower['type']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1 / 30.0
            
            if tower['fire_anim_timer'] > 0:
                tower['fire_anim_timer'] -= 1

            if tower['cooldown'] <= 0:
                target = None
                min_dist = stats['range'] ** 2
                for enemy in self.enemies:
                    dist_sq = (tower['px_pos'][0] - enemy['pos'][0])**2 + (tower['px_pos'][1] - enemy['pos'][1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        'pos': list(tower['px_pos']),
                        'target': target,
                        'stats': stats
                    })
                    tower['cooldown'] = stats['fire_rate']
                    tower['fire_anim_timer'] = 5 # frames
                    # sfx: shoot.wav (different for each tower type)
    
    def _update_projectiles(self):
        kill_reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            dist = math.hypot(target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1])
            if dist < proj['stats']['proj_speed']:
                proj['target']['health'] -= proj['stats']['damage']
                self._create_explosion(proj['pos'], proj['stats']['color'], 5, 1.5)
                self.projectiles.remove(proj)
                # sfx: hit.wav

                if proj['target']['health'] <= 0:
                    self.score += 0.1
                    kill_reward += 0.1
                    self._create_explosion(proj['target']['pos'], self.COLOR_ENEMY, 10, 3)
                    self.enemies.remove(proj['target'])
                    # sfx: enemy_die.wav
            else:
                angle = math.atan2(target_pos[1] - proj_pos[1], target_pos[0] - proj_pos[0])
                proj_pos[0] += math.cos(angle) * proj['stats']['proj_speed']
                proj_pos[1] += math.sin(angle) * proj['stats']['proj_speed']
        return kill_reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            path_idx = enemy['path_index']
            if path_idx >= len(self.path_pixel_coords) - 1:
                self.enemies.remove(enemy)
                self.base_health -= 1
                self.base_hit_timer = 10 # Flash base for 10 frames
                # sfx: base_hit.wav
                continue
            
            target_pos = self.path_pixel_coords[path_idx + 1]
            enemy_pos = enemy['pos']
            
            dist = math.hypot(target_pos[0] - enemy_pos[0], target_pos[1] - enemy_pos[1])
            if dist < enemy['speed']:
                enemy['path_index'] += 1
            else:
                angle = math.atan2(target_pos[1] - enemy_pos[1], target_pos[0] - enemy_pos[0])
                enemy_pos[0] += math.cos(angle) * enemy['speed']
                enemy_pos[1] += math.sin(angle) * enemy['speed']

    def _update_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count, power):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, power)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(10, 20),
                'color': color,
                'size': self.rng.uniform(2, 4)
            })

    def _check_termination(self):
        return self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid and Path
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        if len(self.path_pixel_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixel_coords, self.CELL_SIZE)

        # Base
        base_color = self.COLOR_HEALTH_RED if self.base_hit_timer > 0 else self.COLOR_BASE
        base_rect = pygame.Rect(self.base_pos_grid[0] * self.CELL_SIZE, self.base_pos_grid[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, base_color, base_rect)

        # Towers
        for tower in self.towers:
            stats = self.tower_stats[tower['type']]
            pos = (int(tower['px_pos'][0]), int(tower['px_pos'][1]))
            color = stats['color']
            if tower['fire_anim_timer'] > 0:
                color = (255, 255, 255) # Flash white on fire
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, color)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, (255,255,255), pos, 3)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 16
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos[0] - bar_w/2, pos[1] - 15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0] - bar_w/2, pos[1] - 15, bar_w * health_pct, 3))

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], int(p['size']))

        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = (cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        # Glow effect for cursor
        for i in range(4):
            alpha_color = (*self.COLOR_CURSOR, 50 - i * 10)
            s = pygame.Surface((self.CELL_SIZE + i*2, self.CELL_SIZE + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, alpha_color, s.get_rect(), 0, 3)
            self.screen.blit(s, (cursor_rect[0] - i, cursor_rect[1] - i))

    def _render_ui(self):
        ui_rect = pygame.Rect(self.GAME_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        
        y_offset = 20
        # Wave Info
        self._draw_text(f"Wave: {self.wave_number}/{self.MAX_WAVES}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_medium)
        y_offset += 40
        
        # Resources
        self._draw_text(f"Resources: ${self.resources}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_medium)
        y_offset += 40
        
        # Base Health
        self._draw_text(f"Base HP: {self.base_health}/{self.STARTING_BASE_HEALTH}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_medium)
        y_offset += 40

        # Tower Selection
        self._draw_text("Selected Tower:", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_small)
        y_offset += 25
        
        sel_stats = self.tower_stats[self.selected_tower_type]
        self._draw_text(sel_stats['name'], (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_medium, sel_stats['color'])
        y_offset += 25
        
        self._draw_text(f"Cost: ${sel_stats['cost']}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_small)
        y_offset += 20
        self._draw_text(f"Dmg: {sel_stats['damage']}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_small)
        y_offset += 20
        self._draw_text(f"Rng: {sel_stats['range']}", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_small)
        y_offset += 20
        self._draw_text(f"Rate: {sel_stats['fire_rate']}s", (self.GAME_WIDTH + self.UI_WIDTH / 2, y_offset), self.font_small)
        
        # Start Wave Button
        if self.game_phase == "INTERMISSION":
            button_rect = pygame.Rect(self.GAME_WIDTH + 20, self.SCREEN_HEIGHT - 60, self.UI_WIDTH - 40, 40)
            pygame.draw.rect(self.screen, self.COLOR_BASE, button_rect, 0, 5)
            self._draw_text("START WAVE", button_rect.center, self.font_medium, self.COLOR_BG)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.font_large, color)

    def _draw_text(self, text, pos, font, color=None):
        if color is None:
            color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True

        # --- Human Controls to Action Space Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0
        
        # Buttons
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()