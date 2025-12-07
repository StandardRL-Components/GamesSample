
# Generated: 2025-08-28T04:15:18.875339
# Source Brief: brief_05194.md
# Brief Index: 5194

        
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
        "Controls: Arrow keys to move the cursor. Space to fill/unfill a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A logic puzzle where you fill in squares on a grid to reveal a hidden picture."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_MISTAKES = 5
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID_LINES = (60, 65, 70)
    COLOR_EMPTY = (45, 50, 55)
    COLOR_FILLED_CORRECT = (220, 240, 255)
    COLOR_FILLED_WRONG = (255, 80, 80)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_TARGET_REVEAL = (150, 160, 170)
    COLOR_WIN = (100, 255, 120)
    COLOR_LOSE = (200, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        
        # Fonts
        try:
            self.ui_font = pygame.font.SysFont("Consolas", 24)
            self.end_font = pygame.font.SysFont("Consolas", 72, bold=True)
        except pygame.error:
            self.ui_font = pygame.font.Font(None, 30)
            self.end_font = pygame.font.Font(None, 80)

        # Game state variables are initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mistakes = 0
        self.cursor_pos = [0, 0]
        self.player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.target_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.wrong_squares = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.last_space_press = False
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def _generate_target_image(self):
        """Creates a symmetrical 10x10 target image."""
        half_width = self.GRID_SIZE // 2
        pattern = self.np_random.integers(0, 2, size=(self.GRID_SIZE, half_width), dtype=bool)
        mirrored_pattern = np.fliplr(pattern)
        
        # Ensure there's at least a few squares to fill
        if np.sum(pattern) < 5:
            for _ in range(5):
                pattern[self.np_random.integers(0, self.GRID_SIZE)][self.np_random.integers(0, half_width)] = True
                
        return np.hstack((pattern, mirrored_pattern))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mistakes = 0
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.target_grid = self._generate_target_image()
        self.wrong_squares = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.last_space_press = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0.0

        # --- Handle Input ---
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # Action (toggle square) on space press (rising edge detection)
        space_pressed = space_held and not self.last_space_press
        if space_pressed:
            # SFX: Click.wav
            x, y = self.cursor_pos
            
            # Toggle the player's guess for this square
            self.player_grid[y, x] = not self.player_grid[y, x]

            # Check if this toggle was correct
            is_correct_now = self.player_grid[y, x] == self.target_grid[y, x]

            if is_correct_now:
                # If it was previously marked as wrong, this is a correction
                if self.wrong_squares[y, x]:
                    self.wrong_squares[y, x] = False
                    reward = 1.0  # Reward for fixing a mistake
                    # SFX: Correct.wav
                else: # It was a correct fill
                    reward = 1.0
                    # SFX: Correct.wav
            else: # The toggle resulted in an incorrect state
                # Only penalize and count mistake if it's a new mistake
                if not self.wrong_squares[y, x]:
                    self.mistakes += 1
                    self.wrong_squares[y, x] = True
                    reward = -1.0 # Penalize for making a mistake
                    # SFX: Error.wav
                else: # Toggling an already wrong square back to another wrong state
                    reward = -1.0 # Still a bad move

            self.score += reward

        self.last_space_press = space_held

        # --- Update Game State ---
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        is_complete = np.array_equal(self.player_grid, self.target_grid)

        if self.mistakes >= self.MAX_MISTAKES:
            terminated = True
            self.win = False
            self.game_over = True
            reward -= 100.0  # Large penalty for losing
            self.score -= 100.0
            # SFX: Lose.wav

        elif is_complete:
            terminated = True
            self.win = True
            self.game_over = True
            reward += 100.0  # Large reward for winning
            self.score += 100.0
            # SFX: Win.wav
            
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # No win/loss, just timeout
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Calculate grid dimensions and position to center it
        grid_pixel_size = min(self.SCREEN_HEIGHT - 80, self.SCREEN_WIDTH - 80)
        cell_size = grid_pixel_size // self.GRID_SIZE
        grid_pixel_size = cell_size * self.GRID_SIZE # Recalculate to avoid gaps
        
        grid_top_left_x = (self.SCREEN_WIDTH - grid_pixel_size) // 2
        grid_top_left_y = (self.SCREEN_HEIGHT - grid_pixel_size) // 2

        # Draw squares
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    grid_top_left_x + x * cell_size,
                    grid_top_left_y + y * cell_size,
                    cell_size,
                    cell_size
                )
                
                color = self.COLOR_EMPTY
                if self.game_over:
                    if self.target_grid[y, x]:
                        color = self.COLOR_TARGET_REVEAL
                else:
                    if self.wrong_squares[y, x]:
                        color = self.COLOR_FILLED_WRONG
                    elif self.player_grid[y, x]:
                        color = self.COLOR_FILLED_CORRECT

                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            line_pos_h = grid_top_left_y + i * cell_size
            line_pos_v = grid_top_left_x + i * cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (grid_top_left_x, line_pos_h), (grid_top_left_x + grid_pixel_size, line_pos_h), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (line_pos_v, grid_top_left_y), (line_pos_v, grid_top_left_y + grid_pixel_size), 1)
            
        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(
                grid_top_left_x + self.cursor_pos[0] * cell_size,
                grid_top_left_y + self.cursor_pos[1] * cell_size,
                cell_size,
                cell_size
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score
        score_surf = self.ui_font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Mistakes
        mistakes_surf = self.ui_font.render(f"MISTAKES: {self.mistakes}/{self.MAX_MISTAKES}", True, self.COLOR_UI_TEXT)
        mistakes_rect = mistakes_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(mistakes_surf, mistakes_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg_text = "COMPLETE!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_LOSE
            
            msg_surf = self.end_font.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mistakes": self.mistakes,
            "cursor_pos": self.cursor_pos,
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Picross Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        # Get user input
        movement = 0 # no-op
        space_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            done = False
            
        action = [movement, 1 if space_held else 0, 0] # shift is unused

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    env.close()