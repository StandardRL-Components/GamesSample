
# Generated: 2025-08-27T13:27:10.908832
# Source Brief: brief_00366.md
# Brief Index: 366

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a block. "
        "Doing nothing or placing a block ends your turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base by building a maze. Place blocks to reroute waves of enemies. "
        "Survive 20 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = 20
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_W * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_H * self.CELL_SIZE) // 2
        
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 20
        self.INITIAL_BASE_HEALTH = 100
        self.ENEMY_DAMAGE = 10

        # Cell types
        self.CELL_EMPTY, self.CELL_BLOCK, self.CELL_BASE = 0, 1, 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_BLOCK = (60, 140, 255)
        self.COLOR_BLOCK_FLASH = (200, 220, 255)
        self.COLOR_BASE = (40, 200, 120)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_CURSOR = (255, 220, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_HIGH = (40, 200, 120)
        self.COLOR_HEALTH_MED = (230, 190, 0)
        self.COLOR_HEALTH_LOW = (200, 60, 60)

        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 60, bold=True)
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid and base setup
        self.grid = [[self.CELL_EMPTY for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        self.base_pos = (self.GRID_W // 2, self.GRID_H - 2)
        self.grid[self.base_pos[1]][self.base_pos[0]] = self.CELL_BASE
        
        # Player cursor
        self.cursor_pos = (self.GRID_W // 2, self.GRID_H // 2)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.base_health = self.INITIAL_BASE_HEALTH
        self.game_over = False
        self.game_over_state = ""
        
        # Visual effect timers
        self.block_place_effect_timer = 0
        self.base_hit_effect_timer = 0
        
        # Enemies
        self.enemies = []
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        player_turn_ended = False

        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        # 1. Handle player actions
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.GRID_H - 1, self.cursor_pos[1] + 1))
        elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
        elif movement == 4: self.cursor_pos = (min(self.GRID_W - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

        if space_pressed:
            if self.grid[self.cursor_pos[1]][self.cursor_pos[0]] == self.CELL_EMPTY:
                self.grid[self.cursor_pos[1]][self.cursor_pos[0]] = self.CELL_BLOCK
                reward -= 0.01
                self.block_place_effect_timer = 3  # Visual effect active for 3 renders
                # sfx: block_place.wav
            player_turn_ended = True
        
        if movement == 0 and not space_pressed:
            player_turn_ended = True

        # 2. If player turn ended, advance enemy state
        if player_turn_ended:
            enemies_to_remove = []
            for enemy in self.enemies:
                enemy['move_progress'] += enemy['speed']
                while enemy['move_progress'] >= 1.0:
                    enemy['move_progress'] -= 1.0
                    next_move = self._pathfind_next_step(enemy['pos'], self.base_pos)
                    if next_move:
                        new_pos = (enemy['pos'][0] + next_move[0], enemy['pos'][1] + next_move[1])
                        if new_pos == self.base_pos:
                            self.base_health -= self.ENEMY_DAMAGE
                            self.base_hit_effect_timer = 3
                            enemies_to_remove.append(enemy)
                            # sfx: base_hit.wav
                            break
                        else:
                            enemy['pos'] = new_pos
                            # sfx: enemy_step.wav
                    else:
                        break # No path, enemy is stuck

            if enemies_to_remove:
                self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

            # 3. Check for wave completion
            if not self.enemies:
                # sfx: wave_clear.wav
                reward += 1.0
                reward += self.wave_enemy_count * 0.1
                self.wave += 1
                if self.wave > self.MAX_WAVES:
                    terminated = True
                    reward += 100
                    self.game_over_state = "VICTORY"
                else:
                    self._spawn_wave()

        # 4. Check for termination conditions
        if self.base_health <= 0:
            self.base_health = 0
            if not terminated: # Avoid double penalty
                terminated = True
                reward -= 100
                self.game_over_state = "DEFEAT"
                # sfx: game_over.wav

        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_wave(self):
        num_enemies = 3 + (self.wave - 1) // 3
        enemy_speed = 1.0 + ((self.wave - 1) // 2) * 0.1
        self.wave_enemy_count = num_enemies
        
        spawn_points = [
            (0, 3), (self.GRID_W - 1, 3), (0, 10), (self.GRID_W - 1, 10),
            (self.GRID_W // 4, 0), (self.GRID_W * 3 // 4, 0)
        ]
        
        self.enemies = []
        for _ in range(num_enemies):
            spawn_idx = self.np_random.integers(len(spawn_points))
            spawn_pos = spawn_points[spawn_idx]
            self.enemies.append({
                'pos': spawn_pos,
                'speed': enemy_speed,
                'move_progress': self.np_random.random() # Stagger movement
            })

    def _pathfind_next_step(self, start, end):
        q = collections.deque([(start, [])])
        visited = {start}
        
        while q:
            (cx, cy), path = q.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                if (nx, ny) == end:
                    if not path: return (dx, dy)
                    first_step_pos = path[0]
                    return (first_step_pos[0] - start[0], first_step_pos[1] - start[1])

                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and \
                   self.grid[ny][nx] != self.CELL_BLOCK and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Decrement effect timers
        if self.block_place_effect_timer > 0: self.block_place_effect_timer -= 1
        if self.base_hit_effect_timer > 0: self.base_hit_effect_timer -= 1

        # Draw grid
        for x in range(self.GRID_W + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_H * self.CELL_SIZE))
        for y in range(self.GRID_H + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_W * self.CELL_SIZE, py))

        # Draw grid elements
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                cell = self.grid[y][x]
                rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if cell == self.CELL_BLOCK:
                    pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect.inflate(-2, -2))
                elif cell == self.CELL_BASE:
                    base_color = self.COLOR_BASE
                    if self.base_hit_effect_timer > 0:
                        base_color = self.COLOR_HEALTH_LOW
                    pygame.draw.rect(self.screen, base_color, rect.inflate(-2, -2))
                    # Draw a plus sign on the base
                    cx, cy = rect.center
                    pygame.draw.line(self.screen, self.COLOR_BG, (cx, cy - 4), (cx, cy + 4), 3)
                    pygame.draw.line(self.screen, self.COLOR_BG, (cx - 4, cy), (cx + 4, cy), 3)

        # Draw block placement flash
        if self.block_place_effect_timer > 0:
            x, y = self.cursor_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_FLASH, rect.inflate(-2, -2))

        # Draw enemies
        for enemy in self.enemies:
            x, y = enemy['pos']
            center_x = int(self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_ENEMY)

        # Draw cursor
        if not self.game_over:
            x, y = self.cursor_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))
        
        # Wave
        wave_surf = self.font_ui.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (self.WIDTH - wave_surf.get_width() - 15, 10))
        
        # Base Health Bar
        health_pct = self.base_health / self.INITIAL_BASE_HEALTH if self.INITIAL_BASE_HEALTH > 0 else 0
        health_color = self.COLOR_HEALTH_LOW if health_pct < 0.3 else (self.COLOR_HEALTH_MED if health_pct < 0.6 else self.COLOR_HEALTH_HIGH)
        bar_w, bar_h = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_w) // 2, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_w * health_pct), bar_h))
        
        health_text = self.font_ui.render(f"BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x - health_text.get_width() - 10, 11))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = self.game_over_state
        color = self.COLOR_HEALTH_HIGH if self.game_over_state == "VICTORY" else self.COLOR_HEALTH_LOW
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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