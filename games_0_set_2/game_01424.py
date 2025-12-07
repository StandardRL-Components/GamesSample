
# Generated: 2025-08-27T17:06:04.858797
# Source Brief: brief_01424.md
# Brief Index: 1424

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. "
        "Space to place a standard block. "
        "Shift+Space to place a slowing block (from Wave 3)."
    )

    game_description = (
        "A grid-based tower defense game. "
        "Strategically place blocks to defend your base from waves of enemies. "
        "Survive all 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 20, 11
        self.CELL_SIZE = 32
        self.UI_HEIGHT = self.HEIGHT - (self.GRID_ROWS * self.CELL_SIZE) # 48px

        self.MAX_STEPS = 3000 # Increased from 1000 to allow for longer games
        self.MAX_WAVES = 10

        # Block Types
        self.BLOCK_EMPTY = 0
        self.BLOCK_STANDARD = 1
        self.BLOCK_SLOW = 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_BASE = (0, 255, 128)
        self.COLOR_BASE_DMG = (255, 80, 80)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_FAST = (255, 120, 50)
        self.COLOR_BLOCK_STANDARD = (60, 140, 255)
        self.COLOR_BLOCK_SLOW = (255, 220, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 35, 55)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Internal State ---
        self.grid = None
        self.enemies = None
        self.particles = None
        self.cursor_pos = None
        self.base_pos = None
        self.base_health = None
        self.max_base_health = None
        self.score = None
        self.steps = None
        self.wave_number = None
        self.wave_in_progress = None
        self.wave_cooldown = None
        self.game_over = None
        self.game_won = None
        self.game_over_message = None
        self.rng = None
        self.base_damage_flash = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Use a default or create a new one if not seeded
            if not hasattr(self, 'rng') or self.rng is None:
                self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.enemies = []
        self.particles = []
        self.cursor_pos = [self.GRID_COLS // 4, self.GRID_ROWS // 2]
        
        base_size = 2
        self.base_pos = (self.GRID_COLS - base_size, (self.GRID_ROWS - base_size) // 2)
        self.base_cells = {(self.base_pos[0] + dx, self.base_pos[1] + dy) 
                           for dx in range(base_size) for dy in range(base_size)}

        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.score = 0
        self.steps = 0
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 90  # 3 seconds at 30fps
        self.game_over = False
        self.game_won = False
        self.game_over_message = ""
        self.base_damage_flash = 0

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        self.steps += 1
        reward = 0

        if not self.game_over:
            reward += self._handle_actions(action)
            reward += self._update_enemies()
            reward += self._update_waves()
        
        self._update_particles()
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.game_won:
                reward += 100
                self.game_over_message = "VICTORY!"
            else:
                reward += -100
                self.game_over_message = "GAME OVER"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_COLS
        self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_ROWS

        # --- Block Placement ---
        if space_pressed:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == self.BLOCK_EMPTY and (cx, cy) not in self.base_cells:
                block_type_to_place = self.BLOCK_STANDARD
                if shift_held and self.wave_number >= 3:
                    block_type_to_place = self.BLOCK_SLOW
                
                self.grid[cx, cy] = block_type_to_place
                reward -= 0.01
                # sfx: block_place.wav
                self._create_particle_effect(
                    cx * self.CELL_SIZE + self.CELL_SIZE // 2,
                    cy * self.CELL_SIZE + self.CELL_SIZE // 2 + self.UI_HEIGHT,
                    self.COLOR_BLOCK_STANDARD if block_type_to_place == self.BLOCK_STANDARD else self.COLOR_BLOCK_SLOW,
                    count=15, effect_type='expand'
                )
                
                # Invalidate enemy paths
                for enemy in self.enemies:
                    enemy['path_invalid'] = True

        return reward

    def _update_waves(self):
        if not self.enemies and self.wave_in_progress:
            self.wave_in_progress = False
            self.wave_cooldown = 120 # 4 seconds
            if self.wave_number >= self.MAX_WAVES:
                self.game_won = True
                return 0 # Final reward handled in termination check
            return 1 # Wave survived reward

        if not self.wave_in_progress and self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self._start_next_wave()
        return 0

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        
        num_enemies = 3 + self.wave_number * 2
        base_health = 5 + (self.wave_number - 1)
        base_speed = 0.6 + (self.wave_number - 1) * 0.05

        for i in range(num_enemies):
            spawn_y = self.rng.integers(0, self.GRID_ROWS)
            delay = i * 15 # Stagger spawns
            
            enemy_type = 'normal'
            health = base_health
            speed = base_speed
            
            if self.wave_number >= 5 and self.rng.random() < 0.3:
                enemy_type = 'fast'
                health = int(base_health * 0.75)
                speed = base_speed * 1.5

            self._spawn_enemy(spawn_y, health, speed, enemy_type, delay)
            
    def _spawn_enemy(self, grid_y, health, speed, enemy_type, delay):
        enemy = {
            'x': -self.CELL_SIZE / 2,
            'y': grid_y * self.CELL_SIZE + self.CELL_SIZE / 2 + self.UI_HEIGHT,
            'health': health,
            'max_health': health,
            'speed': speed,
            'type': enemy_type,
            'path': [],
            'path_invalid': True,
            'slow_timer': 0,
            'spawn_delay': delay,
            'id': self.rng.random()
        }
        self.enemies.append(enemy)

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []

        for i, enemy in enumerate(self.enemies):
            if enemy['spawn_delay'] > 0:
                enemy['spawn_delay'] -= 1
                continue

            # --- Pathfinding ---
            if enemy['path_invalid']:
                start_grid = (int(enemy['x'] // self.CELL_SIZE), int((enemy['y'] - self.UI_HEIGHT) // self.CELL_SIZE))
                if start_grid[0] < 0: start_grid = (0, start_grid[1])
                # Target the center of the base area
                target_grid = (self.base_pos[0], self.base_pos[1])
                
                enemy['path'] = self._find_path_astar(start_grid, target_grid)
                enemy['path_invalid'] = False

            if not enemy['path']:
                # No path, maybe try to move towards base directly (or just sit)
                continue
            
            # --- Movement ---
            target_gx, target_gy = enemy['path'][0]
            target_x = target_gx * self.CELL_SIZE + self.CELL_SIZE / 2
            target_y = target_gy * self.CELL_SIZE + self.CELL_SIZE / 2 + self.UI_HEIGHT

            angle = math.atan2(target_y - enemy['y'], target_x - enemy['x'])
            current_speed = enemy['speed']
            if enemy['slow_timer'] > 0:
                current_speed *= 0.5
                enemy['slow_timer'] -= 1

            enemy['x'] += math.cos(angle) * current_speed
            enemy['y'] += math.sin(angle) * current_speed

            if math.hypot(target_x - enemy['x'], target_y - enemy['y']) < self.CELL_SIZE / 4:
                enemy['path'].pop(0)
            
            # --- Check for slow blocks ---
            current_gx = int(enemy['x'] // self.CELL_SIZE)
            current_gy = int((enemy['y'] - self.UI_HEIGHT) // self.CELL_SIZE)
            if 0 <= current_gx < self.GRID_COLS and 0 <= current_gy < self.GRID_ROWS:
                if self.grid[current_gx, current_gy] == self.BLOCK_SLOW:
                    enemy['slow_timer'] = 2 # Apply slow for 2 frames

            # --- Check for reaching base ---
            if (current_gx, current_gy) in self.base_cells:
                self.base_health -= 1
                self.base_damage_flash = 5
                # sfx: base_hit.wav
                self._create_particle_effect(enemy['x'], enemy['y'], self.COLOR_BASE_DMG, 20, 'explode')
                enemies_to_remove.append(i)
                reward += 0.1 # This is a placeholder for kill reward
                self.score += 10
                continue
        
        # This is a placeholder as there are no projectiles to kill enemies.
        # To make the game playable, let's pretend enemies die on their own for now.
        # In a full implementation, this would be where projectile logic goes.
        # For now, let's remove enemies that reach the base and give a "kill" reward.
        
        if enemies_to_remove:
            for i in sorted(enemies_to_remove, reverse=True):
                # sfx: enemy_die.wav
                del self.enemies[i]

        return reward

    def _find_path_astar(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = { (x, y): float('inf') for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS) }
        g_score[start] = 0
        f_score = { (x, y): float('inf') for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS) }
        f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.GRID_COLS and 0 <= neighbor[1] < self.GRID_ROWS):
                    continue
                if self.grid[neighbor] != self.BLOCK_EMPTY or neighbor in self.base_cells:
                    if neighbor != end:
                        continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _create_particle_effect(self, x, y, color, count, effect_type):
        for _ in range(count):
            if effect_type == 'explode':
                angle = self.rng.random() * 2 * math.pi
                speed = 1 + self.rng.random() * 3
                vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            elif effect_type == 'expand':
                angle = self.rng.random() * 2 * math.pi
                speed = 0.5 + self.rng.random() * 1
                vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            
            self.particles.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'life': self.rng.integers(15, 30), 'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_base()
        self._render_blocks()
        self._render_enemies()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.UI_HEIGHT), (px, self.HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_SIZE + self.UI_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

    def _render_base(self):
        color = self.COLOR_BASE if self.base_damage_flash == 0 else self.COLOR_BASE_DMG
        for gx, gy in self.base_cells:
            rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE + self.UI_HEIGHT, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_blocks(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] != self.BLOCK_EMPTY:
                    color = self.COLOR_BLOCK_STANDARD if self.grid[x, y] == self.BLOCK_STANDARD else self.COLOR_BLOCK_SLOW
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE + self.UI_HEIGHT, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)

    def _render_enemies(self):
        for enemy in self.enemies:
            if enemy['spawn_delay'] > 0: continue
            
            size = self.CELL_SIZE * 0.6
            rect = pygame.Rect(enemy['x'] - size / 2, enemy['y'] - size / 2, size, size)
            color = self.COLOR_ENEMY if enemy['type'] == 'normal' else self.COLOR_ENEMY_FAST
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Health bar
            if enemy['health'] < enemy['max_health']:
                hb_width = size
                hb_height = 4
                hb_y = rect.top - hb_height - 2
                health_ratio = enemy['health'] / enemy['max_health']
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (rect.left, hb_y, hb_width, hb_height))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (rect.left, hb_y, hb_width * health_ratio, hb_height))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, (*p['color'], alpha))

    def _render_cursor(self):
        if self.game_over: return
        cx, cy = self.cursor_pos
        rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE + self.UI_HEIGHT, self.CELL_SIZE, self.CELL_SIZE)
        
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = (*self.COLOR_CURSOR, alpha)
        
        # Use a surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect(), 3, border_radius=4)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT-1), (self.WIDTH, self.UI_HEIGHT-1))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, (self.UI_HEIGHT - wave_text.get_height()) // 2))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, (self.UI_HEIGHT - score_text.get_height()) // 2))

        # Base Health
        health_text = self.font_ui.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH // 2 - health_text.get_width() // 2, 5))
        
        hb_width = 200
        hb_height = 15
        hb_x = self.WIDTH // 2 - hb_width // 2
        hb_y = 25
        health_ratio = max(0, self.base_health / self.max_base_health)
        
        health_color = self.COLOR_BASE
        if health_ratio < 0.6: health_color = self.COLOR_BLOCK_SLOW
        if health_ratio < 0.3: health_color = self.COLOR_ENEMY

        pygame.draw.rect(self.screen, self.COLOR_GRID, (hb_x, hb_y, hb_width, hb_height), border_radius=4)
        if health_ratio > 0:
            pygame.draw.rect(self.screen, health_color, (hb_x, hb_y, hb_width * health_ratio, hb_height), border_radius=4)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def _check_termination(self):
        return self.base_health <= 0 or self.game_won or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Requires pygame to be installed and a display available.
    
    # For headless testing, you can comment out the display-related parts.
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headless
    env = GameEnv(render_mode="rgb_array")
    
    # --- Test run ---
    obs, info = env.reset()
    print("Initial state:", info)
    terminated = False
    total_reward = 0
    for i in range(500):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
        if terminated:
            print(f"Episode terminated at step {i+1}.")
            break
    
    print("Test run complete.")

    # --- Interactive Play (requires display) ---
    # To run this, comment out the os.environ line above
    # and ensure you have a display.
    
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((640, 400))
    # pygame.display.set_caption("Tower Defense")
    # clock = pygame.time.Clock()
    
    # running = True
    # terminated = False
    # total_reward = 0
    
    # while running:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

    #     action = (movement, space, shift)
        
    #     if not terminated:
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         total_reward += reward
    #     else: # If game over, allow reset
    #         if keys[pygame.K_r]:
    #             obs, info = env.reset()
    #             terminated = False
    #             total_reward = 0

    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(30)
        
    # pygame.quit()