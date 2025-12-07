
# Generated: 2025-08-27T21:15:14.269143
# Source Brief: brief_02722.md
# Brief Index: 2722

        
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
        "Controls: Arrows to move cursor. Space to place a block. Shift to start the wave."
    )
    game_description = (
        "Defend your fortress by placing blocks to create a maze for incoming enemy waves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 32
        self.GRID_ROWS = 20
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_ROWS
        self.MAX_STEPS = 2000 # Increased from 1000 to allow for more complex scenarios
        self.MAX_WAVES = 10
        self.INITIAL_FORTRESS_HEALTH = 100

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_FORTRESS = (0, 150, 255)
        self.COLOR_FORTRESS_BORDER = (0, 100, 200)
        self.COLOR_SPAWN = (100, 40, 60)
        self.COLOR_BLOCK = (0, 200, 100)
        self.COLOR_BLOCK_BORDER = (0, 150, 75)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_GREEN = (100, 255, 150)
        self.COLOR_PARTICLE_RED = (255, 100, 100)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables (initialized in reset)
        self.game_phase = None
        self.wave_number = None
        self.fortress_health = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.blocks = None
        self.enemies = None
        self.particles = None
        self.cursor_cell = None
        self.fortress_cell = None
        self.spawn_cell = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.fortress_hit_timer = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.game_phase = 'PLACEMENT'
        self.wave_number = 1
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.blocks = set()
        self.enemies = []
        self.particles = []
        
        self.fortress_cell = (self.GRID_COLS - 3, self.GRID_ROWS // 2)
        self.spawn_cell = (2, self.GRID_ROWS // 2)
        self.cursor_cell = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.fortress_hit_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.fortress_hit_timer = max(0, self.fortress_hit_timer - 1)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if self.game_phase == 'PLACEMENT':
            # Handle cursor movement
            if movement == 1: self.cursor_cell = (self.cursor_cell[0], (self.cursor_cell[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS)
            elif movement == 2: self.cursor_cell = (self.cursor_cell[0], (self.cursor_cell[1] + 1) % self.GRID_ROWS)
            elif movement == 3: self.cursor_cell = ((self.cursor_cell[0] - 1 + self.GRID_COLS) % self.GRID_COLS, self.cursor_cell[1])
            elif movement == 4: self.cursor_cell = ((self.cursor_cell[0] + 1) % self.GRID_COLS, self.cursor_cell[1])
            
            # Handle block placement
            if space_press:
                if self.cursor_cell not in self.blocks and self.cursor_cell != self.fortress_cell and self.cursor_cell != self.spawn_cell:
                    self.blocks.add(self.cursor_cell)
                    reward -= 0.01
                    self._create_particles(self.cursor_cell, self.COLOR_PARTICLE_GREEN, 10)
                    # sfx: block_place.wav
            
            # Handle start wave
            if shift_press:
                self.game_phase = 'COMBAT'
                self._spawn_enemies()
                # sfx: wave_start.wav

        elif self.game_phase == 'COMBAT':
            reward += self._update_enemies()
            if not self.enemies and not self.game_over:
                reward += 1.0
                self.score += 10 * self.wave_number
                
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
                    self.win = True
                    reward += 100.0
                    self.score += 1000
                else:
                    self.wave_number += 1
                    self.game_phase = 'PLACEMENT'
                    # sfx: wave_clear.wav

        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.fortress_health <= 0 and not self.win:
            terminated = True
            self.game_over = True
            reward -= 100.0
            # sfx: game_over.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_enemies(self):
        num_enemies = 2 + self.wave_number
        base_speed = 0.02 + (self.wave_number * 0.005)
        for _ in range(num_enemies):
            path = self._find_path(self.spawn_cell, self.fortress_cell)
            if path:
                self.enemies.append({
                    "pos": self._cell_to_pixel(self.spawn_cell),
                    "path": path,
                    "path_index": 0,
                    "speed": base_speed * random.uniform(0.9, 1.1),
                    "progress": 0.0
                })

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy["path_index"] >= len(enemy["path"]) - 1:
                enemies_to_remove.append(i)
                self.fortress_health -= 10
                self.fortress_hit_timer = 10 # Flash for 10 frames
                self.score -= 25
                self._create_particles(self.fortress_cell, self.COLOR_PARTICLE_RED, 20)
                # sfx: fortress_hit.wav
                continue

            enemy["progress"] += enemy["speed"]
            
            while enemy["progress"] >= 1.0 and enemy["path_index"] < len(enemy["path"]) - 1:
                enemy["progress"] -= 1.0
                enemy["path_index"] += 1

            current_cell = enemy["path"][enemy["path_index"]]
            next_cell = enemy["path"][enemy["path_index"] + 1]
            p1 = self._cell_to_pixel(current_cell)
            p2 = self._cell_to_pixel(next_cell)
            
            enemy["pos"] = (
                p1[0] + (p2[0] - p1[0]) * enemy["progress"],
                p1[1] + (p2[1] - p1[1]) * enemy["progress"]
            )
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
            
        return reward

    def _find_path(self, start_cell, end_cell):
        q = deque([(start_cell, [start_cell])])
        visited = {start_cell}
        
        while q:
            current, path = q.popleft()
            if current == end_cell:
                return path
            
            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                
                if (0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and
                        neighbor not in visited and neighbor not in self.blocks):
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return [] # No path found

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw spawn and fortress areas
        spawn_rect = pygame.Rect(self._cell_to_pixel(self.spawn_cell), (self.CELL_WIDTH, self.CELL_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_SPAWN, spawn_rect)
        fortress_rect = pygame.Rect(self._cell_to_pixel(self.fortress_cell), (self.CELL_WIDTH, self.CELL_HEIGHT))
        fortress_color = self.COLOR_FORTRESS if self.fortress_hit_timer == 0 else self.COLOR_ENEMY
        pygame.draw.rect(self.screen, fortress_color, fortress_rect)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS_BORDER, fortress_rect, 2)
        
        # Draw blocks
        for block_cell in self.blocks:
            rect = pygame.Rect(self._cell_to_pixel(block_cell), (self.CELL_WIDTH, self.CELL_HEIGHT))
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_BORDER, rect, 2)
            
        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0] + self.CELL_WIDTH / 2), int(enemy["pos"][1] + self.CELL_HEIGHT / 2))
            radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

        # Draw cursor in placement phase
        if self.game_phase == 'PLACEMENT':
            cursor_rect = pygame.Rect(self._cell_to_pixel(self.cursor_cell), (self.CELL_WIDTH, self.CELL_HEIGHT))
            alpha = 100 + int(math.sin(self.steps * 0.2) * 40)
            surface = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_WIDTH, self.CELL_HEIGHT), 3)
            self.screen.blit(surface, cursor_rect.topleft)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        health_text = self.font_ui.render(f"FORTRESS: {self.fortress_health}%", True, self.COLOR_TEXT)
        phase_text = self.font_ui.render(f"PHASE: {self.game_phase}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        self.screen.blit(health_text, (10, 30))
        self.screen.blit(phase_text, (self.SCREEN_WIDTH - phase_text.get_width() - 10, 30))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.win else "GAME OVER"
            end_text = self.font_game_over.render(end_text_str, True, self.COLOR_CURSOR if self.win else self.COLOR_ENEMY)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _cell_to_pixel(self, cell):
        return (cell[0] * self.CELL_WIDTH, cell[1] * self.CELL_HEIGHT)

    def _create_particles(self, cell, color, count):
        px, py = self._cell_to_pixel(cell)
        center_x = px + self.CELL_WIDTH / 2
        center_y = py + self.CELL_HEIGHT / 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5), 'life': random.randint(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.fortress_health,
            "phase": self.game_phase,
            "blocks": len(self.blocks),
            "win": self.win
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert info['health'] == self.INITIAL_FORTRESS_HEALTH
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # We need a display to see the game and capture keyboard events
    pygame.display.set_caption(GameEnv.game_description)
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # In manual play, we need to decide when to step.
        # For this turn-based game, we step on any key press.
        # A more robust manual player would only step on action changes.
        # For simplicity, we step every frame.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Phase: {info['phase']}")
            
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for manual play
        
    env.close()