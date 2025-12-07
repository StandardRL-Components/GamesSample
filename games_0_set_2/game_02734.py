
# Generated: 2025-08-27T21:16:31.972965
# Source Brief: brief_02734.md
# Brief Index: 2734

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to change a tile's color and trigger a match."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid of colored tiles by strategically changing tile colors to create adjacent groups of 3 or more. Clear the board before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 8
    GRID_TOP_MARGIN = 60
    GRID_SIDE_MARGIN = 140
    TILE_SIZE = (SCREEN_WIDTH - 2 * GRID_SIDE_MARGIN) // GRID_COLS
    
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60-second game

    # Colors (bright, distinct, colorblind-friendly)
    COLOR_BG = (15, 23, 42)
    COLOR_GRID_LINES = (51, 65, 85)
    COLOR_EMPTY = (30, 41, 59)
    COLOR_TEXT = (226, 232, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TIMER_BAR = (34, 197, 94)
    COLOR_TIMER_BAR_WARN = (234, 179, 8)
    COLOR_TIMER_BAR_DANGER = (220, 38, 38)
    
    TILE_COLORS = [
        (59, 130, 246),  # Blue
        (239, 68, 68),   # Red
        (34, 197, 94),   # Green
        (249, 115, 22),  # Orange
        (168, 85, 247),  # Purple
        (236, 72, 153),  # Pink
        (14, 165, 233),  # Sky
        (245, 208, 254), # Fuchsia
    ]
    NUM_COLORS = len(TILE_COLORS)
    EMPTY_TILE = -1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 20)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.timer = None
        self.game_over = None
        self.space_was_held = None
        self.particles = None
        
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.space_was_held = True # Prevent action on first frame
        self.particles = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_ROWS, self.GRID_COLS))

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.timer -= 1

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        self._handle_movement(movement)
        
        clicked = space_held and not self.space_was_held
        if clicked:
            reward += self._handle_click()
            
        self.space_was_held = space_held

        # --- Update Game State ---
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self.timer <= 0 or self._is_grid_clear() or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_grid_clear():
                reward += 100 # Win bonus
                # Sound: 'win_game'
            else:
                reward -= 100 # Lose penalty
                # Sound: 'lose_game'

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

    def _handle_click(self):
        cx, cy = self.cursor_pos
        current_tile = self.grid[cy][cx]
        
        if current_tile != self.EMPTY_TILE:
            # Sound: 'tile_change'
            # Cycle color
            self.grid[cy][cx] = (current_tile + 1) % self.NUM_COLORS
            new_color_idx = self.grid[cy][cx]
            
            # Find matches for the new color
            matches = self._find_matches(cx, cy, new_color_idx)
            
            if len(matches) >= 3:
                # Sound: 'match_clear'
                for pos_x, pos_y in matches:
                    self._spawn_particles(pos_x, pos_y, self.TILE_COLORS[new_color_idx])
                    self.grid[pos_y][pos_x] = self.EMPTY_TILE
                
                self._apply_gravity_and_refill()
                
                # Calculate reward
                num_cleared = len(matches)
                self.score += num_cleared
                # Reward for clearing tiles + bonus for larger chains
                return (num_cleared * 0.1) + max(0, num_cleared - 2)
        return 0

    def _find_matches(self, start_x, start_y, target_color_idx):
        if not (0 <= start_x < self.GRID_COLS and 0 <= start_y < self.GRID_ROWS):
            return []
            
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        match_group = []

        while q:
            x, y = q.popleft()
            
            if self.grid[y][x] == target_color_idx:
                match_group.append((x,y))
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return match_group

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != self.EMPTY_TILE:
                    if r != write_row:
                        self.grid[write_row][c] = self.grid[r][c]
                        self.grid[r][c] = self.EMPTY_TILE
                    write_row -= 1
            # Refill empty top cells
            for r in range(write_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, self.NUM_COLORS)

    def _is_grid_clear(self):
        return np.all(self.grid == self.EMPTY_TILE)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps_remaining": self.timer,
        }

    def _render_game(self):
        # Draw grid and tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.grid[r][c]
                rect = pygame.Rect(
                    self.GRID_SIDE_MARGIN + c * self.TILE_SIZE,
                    self.GRID_TOP_MARGIN + r * self.TILE_SIZE,
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )
                
                color = self.COLOR_EMPTY if tile_val == self.EMPTY_TILE else self.TILE_COLORS[tile_val]
                
                # Draw tile with a border effect
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect)
                inner_rect = rect.inflate(-4, -4)
                pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
                
        # Draw particles
        for p in self.particles:
            p_color = list(p['color'])
            p_color_with_alpha = (*p_color, p['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), p_color_with_alpha)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_SIDE_MARGIN + self.cursor_pos[0] * self.TILE_SIZE,
            self.GRID_TOP_MARGIN + self.cursor_pos[1] * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        # Use gfxdraw for smooth, thick, anti-aliased lines
        for i in range(3):
            pygame.gfxdraw.rectangle(self.screen, cursor_rect.inflate(i, i), (*self.COLOR_CURSOR, 150-i*40))

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Render timer bar
        timer_percent = max(0, self.timer / self.MAX_STEPS)
        bar_width = 200
        bar_height = 25
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        bar_color = self.COLOR_TIMER_BAR
        if timer_percent < 0.5: bar_color = self.COLOR_TIMER_BAR_WARN
        if timer_percent < 0.2: bar_color = self.COLOR_TIMER_BAR_DANGER
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width * timer_percent, bar_height), border_radius=5)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "BOARD CLEARED!" if self._is_grid_clear() else "TIME'S UP!"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_SIDE_MARGIN + grid_x * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.GRID_TOP_MARGIN + grid_y * self.TILE_SIZE + self.TILE_SIZE / 2
        
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5),
                'alpha': 255,
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['alpha'] -= 10
            p['size'] *= 0.98
        self.particles = [p for p in self.particles if p['alpha'] > 0]
        
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

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # Switch to a windowed render mode
    env.metadata["render_modes"].append("human")
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Store key presses
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }

    while not terminated:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render to the screen
        screen_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(screen_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before closing
            
    env.close()