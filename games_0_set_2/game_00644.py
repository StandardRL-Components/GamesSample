
# Generated: 2025-08-27T14:18:46.703224
# Source Brief: brief_00644.md
# Brief Index: 644

        
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
        "This is a turn-based puzzle. Use arrow keys to push the selected pixel. Each push costs 1 move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image by pushing pixels into their correct places. Correctly placed pixels regain their color. Solve the puzzle before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000 # Safety limit

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80) # Dark blue-gray
        self.COLOR_GRID_LINES = (52, 73, 94)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_SELECTOR = (241, 196, 15) # Yellow
        self.COLOR_WIN = (46, 204, 113, 200) # Green, semi-transparent
        self.COLOR_LOSE = (231, 76, 60, 200) # Red, semi-transparent
        
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Rendering constants
        self.GRID_AREA_SIZE = 320
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_TOP_LEFT = (
            (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) // 2
        )
        
        # Initialize state variables
        self.target_grid = None
        self.pixel_grid = None
        self.selector_pos = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._create_target_image()
        
        # Create shuffled pixel grid
        pixels = [self.target_grid[y, x] for y in range(self.GRID_SIZE) for x in range(self.GRID_SIZE)]
        self.np_random.shuffle(pixels)
        self.pixel_grid = np.array(pixels).reshape((self.GRID_SIZE, self.GRID_SIZE, 3))
        
        # Initialize game state
        self.selector_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def _create_target_image(self):
        # Procedurally generate a 10x10 smiley face
        self.target_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE, 3), dtype=np.uint8)
        
        # Yellow face
        face_color = (255, 225, 25)
        self.target_grid[:, :] = face_color

        # Black background for features
        bg_color = (20, 20, 20)
        self.target_grid[0, :] = bg_color
        self.target_grid[9, :] = bg_color
        self.target_grid[:, 0] = bg_color
        self.target_grid[:, 9] = bg_color
        self.target_grid[1, 1:9] = bg_color
        self.target_grid[8, 1:9] = bg_color
        self.target_grid[2:8, 1] = bg_color
        self.target_grid[2:8, 8] = bg_color

        # Eyes
        eye_color = (20, 20, 20)
        self.target_grid[3, 3] = eye_color
        self.target_grid[3, 6] = eye_color

        # Mouth
        mouth_color = (20, 20, 20)
        self.target_grid[6, 3:7] = mouth_color
        self.target_grid[7, 4:6] = mouth_color

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing and return final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        self.steps += 1
        reward = 0

        movement = action[0]  # 0-4: none/up/down/left/right
        
        if movement > 0: # An actual move is attempted
            self.moves_left -= 1
            
            # Store pre-move state for reward calculation
            grid_before_swap = self.pixel_grid.copy()
            x, y = self.selector_pos
            was_p1_correct = np.array_equal(self.pixel_grid[y, x], self.target_grid[y, x])
            
            # Determine target position for the swap
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            nx, ny = (x + dx) % self.GRID_SIZE, (y + dy) % self.GRID_SIZE
            was_p2_correct = np.array_equal(self.pixel_grid[ny, nx], self.target_grid[ny, nx])

            # Perform the swap
            p1_color = self.pixel_grid[y, x].copy()
            self.pixel_grid[y, x] = self.pixel_grid[ny, nx]
            self.pixel_grid[ny, nx] = p1_color

            # Update selector position to follow the pushed pixel
            self.selector_pos = (nx, ny)

            # Calculate reward based on the swap
            is_p1_now_correct = np.array_equal(self.pixel_grid[y, x], self.target_grid[y, x])
            is_p2_now_correct = np.array_equal(self.pixel_grid[ny, nx], self.target_grid[ny, nx])
            
            # Reward for the pixel that moved into (y,x)
            if not was_p2_correct and is_p1_now_correct: reward += 1.0
            elif was_p2_correct and not is_p1_now_correct: reward -= 0.2
            
            # Reward for the pixel that moved into (ny,nx)
            if not was_p1_correct and is_p2_now_correct: reward += 1.0
            elif was_p1_correct and not is_p2_now_correct: reward -= 0.2

            # Reward for completing rows/columns
            reward += self._calculate_completion_reward(grid_before_swap)

        # Check for termination conditions
        is_win = np.array_equal(self.pixel_grid, self.target_grid)
        is_loss = self.moves_left <= 0 and not is_win
        is_timeout = self.steps >= self.MAX_STEPS

        terminated = is_win or is_loss or is_timeout
        if terminated:
            self.game_over = True
            if is_win:
                reward += 100
            else: # Loss or timeout
                reward -= 50
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_completion_reward(self, grid_before):
        reward = 0
        for i in range(self.GRID_SIZE):
            # Check rows
            was_row_complete = np.array_equal(grid_before[i, :], self.target_grid[i, :])
            is_row_complete = np.array_equal(self.pixel_grid[i, :], self.target_grid[i, :])
            if not was_row_complete and is_row_complete:
                reward += 5

            # Check columns
            was_col_complete = np.array_equal(grid_before[:, i], self.target_grid[:, i])
            is_col_complete = np.array_equal(self.pixel_grid[:, i], self.target_grid[:, i])
            if not was_col_complete and is_col_complete:
                reward += 5
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.GRID_TOP_LEFT[0] + x * self.CELL_SIZE,
                    self.GRID_TOP_LEFT[1] + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                pixel_color = self.pixel_grid[y, x]
                target_color = self.target_grid[y, x]

                is_correct = np.array_equal(pixel_color, target_color)
                
                if is_correct:
                    draw_color = tuple(pixel_color)
                else:
                    # Render as grayscale if incorrect
                    gray_val = int(sum(pixel_color) / 3)
                    draw_color = (gray_val, gray_val, gray_val)
                
                pygame.draw.rect(self.screen, draw_color, cell_rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_TOP_LEFT[0] + i * self.CELL_SIZE, self.GRID_TOP_LEFT[1])
            end_pos = (self.GRID_TOP_LEFT[0] + i * self.CELL_SIZE, self.GRID_TOP_LEFT[1] + self.GRID_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_TOP_LEFT[0], self.GRID_TOP_LEFT[1] + i * self.CELL_SIZE)
            end_pos = (self.GRID_TOP_LEFT[0] + self.GRID_AREA_SIZE, self.GRID_TOP_LEFT[1] + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)
        
        # Draw selector
        sel_x, sel_y = self.selector_pos
        selector_rect = pygame.Rect(
            self.GRID_TOP_LEFT[0] + sel_x * self.CELL_SIZE,
            self.GRID_TOP_LEFT[1] + sel_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Pulsing effect for selector border
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
        width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width)

    def _render_ui(self):
        # --- Left Panel: Info ---
        moves_text = self.font_main.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))

        # --- Right Panel: Target Preview ---
        preview_size = 100
        preview_cell_size = preview_size // self.GRID_SIZE
        preview_top_left = (self.SCREEN_WIDTH - preview_size - 20, 20)
        
        preview_title = self.font_title.render("Target Image", True, self.COLOR_TEXT)
        self.screen.blit(preview_title, (preview_top_left[0], preview_top_left[1]))

        preview_area_rect = pygame.Rect(preview_top_left[0], preview_top_left[1] + 25, preview_size, preview_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, preview_area_rect, 1)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = tuple(self.target_grid[y, x])
                rect = pygame.Rect(
                    preview_area_rect.left + x * preview_cell_size,
                    preview_area_rect.top + y * preview_cell_size,
                    preview_cell_size, preview_cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # --- Game Over Overlay ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            is_win = np.array_equal(self.pixel_grid, self.target_grid)
            
            if is_win:
                overlay.fill(self.COLOR_WIN)
                text = self.font_game_over.render("YOU WIN!", True, self.COLOR_TEXT)
            else:
                overlay.fill(self.COLOR_LOSE)
                text = self.font_game_over.render("OUT OF MOVES", True, self.COLOR_TEXT)
                
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Sort")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                
                # Only step if a movement key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
    
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()