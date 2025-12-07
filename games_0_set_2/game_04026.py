
# Generated: 2025-08-28T01:09:17.634746
# Source Brief: brief_04026.md
# Brief Index: 4026

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through colors. Press Space to fill a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Uncover a hidden pixel art image by strategically filling in a grid. You have a limited number of moves, so choose your colors and placements wisely!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID_LINE = (50, 50, 60)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAILURE = (255, 100, 100)
    
    PALETTE = [
        (40, 40, 50),       # 0: Empty/Background
        (255, 80, 80),      # 1: Red
        (80, 150, 255),     # 2: Blue
        (80, 255, 150),     # 3: Green
        (255, 220, 80)      # 4: Yellow
    ]
    PALETTE_DESATURATED = [
        (40, 40, 50),
        (120, 60, 60),
        (60, 90, 120),
        (60, 120, 90),
        (120, 110, 60)
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Grid layout
        self.cell_size = 32
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        # Initialize state variables
        self.target_image = self._create_target_image()
        self.total_pixels = np.count_nonzero(self.target_image)
        self.current_grid = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.remaining_moves = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def _create_target_image(self):
        # A simple heart shape using colors 1 (red) and 2 (blue)
        # 0: BG, 1: Outline (Red), 2: Fill (Blue)
        img = np.array([
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 2, 2, 1, 1, 2, 2, 1, 0],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 0, 1, 2, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        return img
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.current_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.cursor_pos = np.array([0, 0])
        self.selected_color_idx = 1 # Start with the first paintable color
        self.remaining_moves = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        if not self.game_over:
            # --- Handle Actions ---
            # 1. Cursor Movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

            # 2. Cycle Color (on key press)
            if shift_held and not self.prev_shift_held:
                self.selected_color_idx = (self.selected_color_idx % 4) + 1 # Cycle through 1,2,3,4
                # Sfx: UI_Cycle.wav

            # 3. Place Color (on key press)
            if space_held and not self.prev_space_held and self.remaining_moves > 0:
                reward += self._place_color()
                # Sfx: Place_Pixel.wav
        
        # Update button press states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Update particle animations
        self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        is_complete = np.array_equal(self.current_grid, self.target_image)
        is_out_of_moves = self.remaining_moves <= 0
        is_max_steps = self.steps >= self.MAX_STEPS
        
        terminated = is_complete or is_out_of_moves or is_max_steps

        if terminated and not self.game_over:
            self.game_over = True
            if is_complete:
                reward += 50  # Goal-oriented reward for winning
                # Sfx: Win_Jingle.wav
            else:
                reward -= 50  # Goal-oriented penalty for losing
                # Sfx: Lose_Sound.wav

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_color(self):
        self.remaining_moves -= 1
        px, py = self.cursor_pos
        
        # Can't overwrite an already correct pixel
        if self.current_grid[py, px] == self.target_image[py, px] and self.target_image[py, px] != 0:
            return -0.5 # Penalty for wasting a move

        placed_color = self.selected_color_idx
        target_color = self.target_image[py, px]
        
        # Check for row/col completion *before* changing the grid
        old_row = self.current_grid[py, :].copy()
        old_col = self.current_grid[:, px].copy()

        self.current_grid[py, px] = placed_color
        
        # Create particle effect
        particle_x = self.grid_x + px * self.cell_size + self.cell_size // 2
        particle_y = self.grid_y + py * self.cell_size + self.cell_size // 2
        self.particles.append([particle_x, particle_y, self.cell_size * 0.5, 15]) # x, y, radius, lifetime
        
        # --- Calculate Reward ---
        reward = 0
        # 1. Pixel placement reward
        if placed_color == target_color:
            reward += 1.0
        else:
            reward -= 0.1
        
        # 2. Row/Column completion reward
        new_row = self.current_grid[py, :]
        new_col = self.current_grid[:, px]
        target_row = self.target_image[py, :]
        target_col = self.target_image[:, px]

        # Check if row is now complete and wasn't before
        if np.array_equal(new_row, target_row) and not np.array_equal(old_row, target_row):
            reward += 5.0
        # Check if column is now complete and wasn't before
        if np.array_equal(new_col, target_col) and not np.array_equal(old_col, target_col):
            reward += 5.0

        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # 1. Draw faint target image as a hint
        hint_surface = pygame.Surface((self.grid_width, self.grid_height), pygame.SRCALPHA)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.target_image[y, x]
                if color_idx > 0:
                    color = self.PALETTE[color_idx]
                    pygame.draw.rect(hint_surface, (*color, 30), 
                                     (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        self.screen.blit(hint_surface, (self.grid_x, self.grid_y))

        # 2. Draw placed pixels
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                placed_idx = self.current_grid[y, x]
                if placed_idx > 0:
                    target_idx = self.target_image[y, x]
                    color = self.PALETTE[placed_idx] if placed_idx == target_idx else self.PALETTE_DESATURATED[placed_idx]
                    rect = (self.grid_x + x * self.cell_size, self.grid_y + y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, color, rect)

        # 3. Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_x + i * self.cell_size, self.grid_y)
            end_pos = (self.grid_x + i * self.cell_size, self.grid_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.grid_x, self.grid_y + i * self.cell_size)
            end_pos = (self.grid_x + self.grid_width, self.grid_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            
        # 4. Draw particles
        for p in self.particles:
            alpha = int(255 * (p[3] / 15.0))
            color = (*self.COLOR_CURSOR, alpha)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), int(p[2]), color)
            
        # 5. Draw cursor
        if not self.game_over:
            cursor_x = self.grid_x + self.cursor_pos[0] * self.cell_size
            cursor_y = self.grid_y + self.cursor_pos[1] * self.cell_size
            
            # Pulsing alpha for the cursor fill
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
            alpha = 30 + pulse * 40
            
            cursor_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            cursor_surface.fill((*self.COLOR_CURSOR, alpha))
            self.screen.blit(cursor_surface, (cursor_x, cursor_y))
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.cell_size, self.cell_size), 2)
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p[3] > 0]
        for p in self.particles:
            p[2] += 0.5  # Expand radius
            p[3] -= 1    # Decrease lifetime

    def _render_ui(self):
        # 1. Moves Left
        moves_text = self.font_main.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # 2. Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))
        
        # 3. Color Palette
        palette_y = self.SCREEN_HEIGHT - 40
        for i in range(1, len(self.PALETTE)):
            color = self.PALETTE[i]
            x_pos = self.SCREEN_WIDTH // 2 - (4 * 30) // 2 + (i-1) * 30
            rect = (x_pos, palette_y, 25, 25)
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_idx and not self.game_over:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
                
        # 4. Progress Bar
        correct_pixels = np.sum((self.current_grid == self.target_image) & (self.target_image != 0))
        progress = correct_pixels / self.total_pixels if self.total_pixels > 0 else 1.0
        
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 25
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SUCCESS, (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        # 5. Game Over Message
        if self.game_over:
            is_complete = np.array_equal(self.current_grid, self.target_image)
            msg = "COMPLETE!" if is_complete else "GAME OVER"
            color = self.COLOR_SUCCESS if is_complete else self.COLOR_FAILURE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "cursor_pos": self.cursor_pos.tolist(),
        }

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
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # Default no-op
        keys = pygame.key.get_pressed()

        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Other actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}")
            # In a real play session, you might want to wait for a reset key press
            # For this example, we just let the loop continue showing the final screen
    
    env.close()