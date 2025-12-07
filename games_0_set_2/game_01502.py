
# Generated: 2025-08-27T17:21:06.664606
# Source Brief: brief_01502.md
# Brief Index: 1502

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Press Space to paint. Press Shift to cycle color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art pattern before the timer runs out. Move your cursor, "
        "select the right color, and paint each pixel to match the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game Constants ---
        self.GRID_SIZE = 10
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 30
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_EMPTY = (60, 65, 78)
        self.COLOR_GRID_LINE = (40, 43, 54)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (138, 245, 138)
        self.COLOR_FAIL = (245, 138, 138)
        
        self.PAINT_COLORS = [
            (227, 95, 81),   # Red
            (99, 193, 82),   # Green
            (78, 148, 219),  # Blue
        ]
        
        # --- Layout ---
        self.PLAYER_CELL_SIZE = 32
        self.PLAYER_GRID_MARGIN = 2
        player_grid_dim = self.GRID_SIZE * (self.PLAYER_CELL_SIZE + self.PLAYER_GRID_MARGIN)
        self.player_grid_pos = (
            (self.screen_width - player_grid_dim) // 2,
            (self.screen_height - player_grid_dim) // 2 + 20
        )
        
        self.TARGET_CELL_SIZE = 10
        self.TARGET_GRID_MARGIN = 1
        target_grid_dim = self.GRID_SIZE * (self.TARGET_CELL_SIZE + self.TARGET_GRID_MARGIN)
        self.target_grid_pos = ((self.screen_width - target_grid_dim) // 2, 15)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.cursor_pos = None
        self.current_color_index = 0
        self.player_grid = None
        self.target_grid = None
        self.last_space_pressed = False
        self.last_shift_pressed = False
        self.rewarded_rows = None
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.current_color_index = 0
        
        # Generate a new random target pattern
        self.target_grid = self.np_random.integers(0, len(self.PAINT_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
        
        # Player's grid is empty (-1 represents the empty color)
        self.player_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)
        
        self.last_space_pressed = False
        self.last_shift_pressed = False
        self.rewarded_rows = [False] * self.GRID_SIZE
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        # 2. Handle Color Cycle (on press, not hold)
        if shift_held and not self.last_shift_pressed:
            self.current_color_index = (self.current_color_index + 1) % len(self.PAINT_COLORS)
            # sfx: color_cycle_sound

        # 3. Handle Painting (on press, not hold)
        if space_held and not self.last_space_pressed:
            y, x = self.cursor_pos[1], self.cursor_pos[0]
            
            # Only apply changes if the new color is different
            if self.player_grid[y, x] != self.current_color_index:
                self.player_grid[y, x] = self.current_color_index
                target_color = self.target_grid[y, x]
                
                if self.current_color_index == target_color:
                    reward += 1.0
                    self._add_particles(self.cursor_pos, self.PAINT_COLORS[self.current_color_index])
                    # sfx: correct_paint_sound
                else:
                    reward -= 0.2
                    # sfx: wrong_paint_sound

                # Check for row completion bonus
                if not self.rewarded_rows[y] and np.array_equal(self.player_grid[y], self.target_grid[y]):
                    reward += 5.0
                    self.rewarded_rows[y] = True
                    # sfx: row_complete_sound

        # 4. Update timer and game state
        self.timer -= 1
        self.steps += 1
        
        # 5. Check for termination
        is_complete = np.array_equal(self.player_grid, self.target_grid)
        time_out = self.timer <= 0
        terminated = is_complete or time_out
        
        if terminated and not self.game_over:
            self.game_over = True
            if is_complete:
                reward += 100.0  # Big bonus for winning
                # sfx: win_jingle
            else: # Time out
                reward -= 50.0  # Big penalty for losing
                # sfx: lose_fanfare

        self.score += reward
        self.last_space_pressed = space_held
        self.last_shift_pressed = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Update and render particles
        self._update_and_draw_particles()
        
        # Render game elements
        self._render_grids()
        self._render_cursor()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def _render_grids(self):
        # Render Target Grid
        self._render_grid(self.target_grid, self.target_grid_pos, self.TARGET_CELL_SIZE, self.TARGET_GRID_MARGIN)
        
        # Render Player Grid
        self._render_grid(self.player_grid, self.player_grid_pos, self.PLAYER_CELL_SIZE, self.PLAYER_GRID_MARGIN)

    def _render_grid(self, grid_data, pos, cell_size, margin):
        start_x, start_y = pos
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_index = grid_data[y, x]
                color = self.PAINT_COLORS[color_index] if color_index != -1 else self.COLOR_EMPTY
                
                rect = pygame.Rect(
                    start_x + x * (cell_size + margin),
                    start_y + y * (cell_size + margin),
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def _render_cursor(self):
        if self.game_over: return
        
        # Pulsating effect for cursor highlight
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # Varies between 0 and 1
        thickness = 2 + int(pulse * 2)

        px, py = self.player_grid_pos
        cs, m = self.PLAYER_CELL_SIZE, self.PLAYER_GRID_MARGIN
        cx, cy = self.cursor_pos

        cursor_rect = pygame.Rect(
            px + cx * (cs + m) - thickness,
            py + cy * (cs + m) - thickness,
            cs + thickness * 2,
            cs + thickness * 2
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, thickness, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT if time_left > 5 else self.COLOR_FAIL
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(timer_text, timer_rect)
        
        # Current Color Indicator
        indicator_size = 40
        indicator_rect = pygame.Rect(
            (self.screen_width - indicator_size) // 2, 
            self.screen_height - indicator_size - 10,
            indicator_size, 
            indicator_size
        )
        current_color = self.PAINT_COLORS[self.current_color_index]
        pygame.draw.rect(self.screen, current_color, indicator_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, indicator_rect, 2, border_radius=5)

        # Game Over Message
        if self.game_over:
            is_complete = np.array_equal(self.player_grid, self.target_grid)
            msg = "PERFECT!" if is_complete else "TIME'S UP!"
            color = self.COLOR_SUCCESS if is_complete else self.COLOR_FAIL
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)
            
    def _add_particles(self, grid_pos, color):
        px, py = self.player_grid_pos
        cs, m = self.PLAYER_CELL_SIZE, self.PLAYER_GRID_MARGIN
        
        center_x = px + grid_pos[0] * (cs + m) + cs // 2
        center_y = py + grid_pos[1] * (cs + m) + cs // 2

        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'radius': random.uniform(2, 5),
                'color': color,
                'life': random.randint(10, 20)
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] -= 0.1
            p['life'] -= 1
            if p['life'] > 0 and p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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