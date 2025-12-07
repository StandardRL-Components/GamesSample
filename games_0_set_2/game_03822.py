
# Generated: 2025-08-28T00:32:27.444356
# Source Brief: brief_03822.md
# Brief Index: 3822

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to push the selected column UP. Press Shift to push the selected row LEFT."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image by pushing rows and columns of pixels. Plan your moves carefully before you run out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    MAX_MOVES = 20
    
    # Colors
    COLOR_BG = (29, 43, 83)  # Dark Blue
    COLOR_GRID_BG = (0, 0, 0)
    COLOR_GRID_LINE = (41, 60, 115)
    COLOR_TEXT = (255, 241, 232)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_HIGHLIGHT = (255, 204, 170) # Peach
    
    # Pixel Palette (PICO-8 inspired)
    PALETTE = [
        (131, 118, 156), # Purple
        (255, 0, 77),    # Red
        (0, 135, 81),    # Green
        (0, 228, 54),    # Bright Green
        (41, 173, 255),  # Blue
        (255, 163, 0),   # Orange
        (255, 236, 39),  # Yellow
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("consolas", 64, bold=True)
        self.font_win = pygame.font.SysFont("consolas", 32, bold=True)

        self.render_mode = render_mode
        self.grid_pixel_size = 0
        self.playable_grid_rect = None
        self.target_grid_rect = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        
        # Puzzle Generation
        self.target_grid = self.np_random.integers(0, len(self.PALETTE), size=(self.GRID_SIZE, self.GRID_SIZE))
        
        # Scramble the target grid to create a solvable puzzle
        scrambled_grid = np.copy(self.target_grid)
        num_scrambles = self.np_random.integers(10, 20)
        for _ in range(num_scrambles):
            is_col = self.np_random.choice([True, False])
            index = self.np_random.integers(0, self.GRID_SIZE)
            if is_col:
                scrambled_grid[:, index] = np.roll(scrambled_grid[:, index], 1)
            else:
                scrambled_grid[index, :] = np.roll(scrambled_grid[index, :], 1)
        self.playable_grid = scrambled_grid

        # Player State
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        
        # UI/FX State
        self.last_push_info = None # {'type': 'row'/'col', 'index': int}

        # Initial Score Calculation
        self.score = self._calculate_correct_pixels(self.playable_grid)
        self.max_correct_pixels = self.score
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.last_push_info = None
        reward = 0
        
        # 1. Handle Cursor Movement
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # 2. Handle Push Actions (Space/Shift)
        pushed = False
        if space_pressed: # Push column UP
            pushed = True
            self.last_push_info = {'type': 'col', 'index': self.cursor_pos[0]}
            # # Sound: "ui_push_vertical.wav"
            
            old_score = self.score
            
            # Perform push
            col_idx = self.cursor_pos[0]
            self.playable_grid[:, col_idx] = np.roll(self.playable_grid[:, col_idx], -1)

            # Calculate new score and reward
            new_score = self._calculate_correct_pixels(self.playable_grid)
            reward = new_score - old_score

            if new_score > self.max_correct_pixels:
                reward += 5 # New high score bonus
                self.max_correct_pixels = new_score
            elif new_score <= old_score:
                reward -= 0.2 * old_score

            self.score = new_score

        elif shift_pressed: # Push row LEFT
            pushed = True
            self.last_push_info = {'type': 'row', 'index': self.cursor_pos[1]}
            # # Sound: "ui_push_horizontal.wav"

            old_score = self.score
            
            # Perform push
            row_idx = self.cursor_pos[1]
            self.playable_grid[row_idx, :] = np.roll(self.playable_grid[row_idx, :], -1)

            # Calculate new score and reward
            new_score = self._calculate_correct_pixels(self.playable_grid)
            reward = new_score - old_score

            if new_score > self.max_correct_pixels:
                reward += 5 # New high score bonus
                self.max_correct_pixels = new_score
            elif new_score <= old_score:
                reward -= 0.2 * old_score

            self.score = new_score

        if pushed:
            self.moves_remaining -= 1

        # 3. Check for Termination
        terminated = False
        perfect_match = np.array_equal(self.playable_grid, self.target_grid)

        if perfect_match:
            reward += 100
            terminated = True
            # # Sound: "win_puzzle.wav"
        elif self.moves_remaining <= 0:
            terminated = True
            # # Sound: "lose_puzzle.wav"
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _calculate_correct_pixels(self, grid):
        return np.sum(grid == self.target_grid)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "correct_pixels": self.score,
            "max_correct_pixels": self.max_correct_pixels,
        }

    def _render_grid(self, surface, grid_data, top_left_pos, size, title):
        grid_width = size
        self.grid_pixel_size = grid_width // self.GRID_SIZE
        
        # Draw grid background
        grid_rect = pygame.Rect(top_left_pos[0], top_left_pos[1], grid_width, grid_width)
        pygame.draw.rect(surface, self.COLOR_GRID_BG, grid_rect)
        
        # Draw pixels
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = grid_data[r, c]
                color = self.PALETTE[color_index]
                pixel_rect = pygame.Rect(
                    top_left_pos[0] + c * self.grid_pixel_size,
                    top_left_pos[1] + r * self.grid_pixel_size,
                    self.grid_pixel_size,
                    self.grid_pixel_size
                )
                pygame.draw.rect(surface, color, pixel_rect)
        
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (top_left_pos[0] + i * self.grid_pixel_size, top_left_pos[1])
            end_pos = (top_left_pos[0] + i * self.grid_pixel_size, top_left_pos[1] + grid_width)
            pygame.draw.line(surface, self.COLOR_GRID_LINE, start_pos, end_pos)
            # Horizontal
            start_pos = (top_left_pos[0], top_left_pos[1] + i * self.grid_pixel_size)
            end_pos = (top_left_pos[0] + grid_width, top_left_pos[1] + i * self.grid_pixel_size)
            pygame.draw.line(surface, self.COLOR_GRID_LINE, start_pos, end_pos)

        # Draw title
        title_surf = self.font_title.render(title, True, self.COLOR_TEXT)
        title_pos = (grid_rect.centerx - title_surf.get_width() // 2, grid_rect.top - title_surf.get_height() - 5)
        surface.blit(title_surf, title_pos)

        return grid_rect

    def _render_game(self):
        grid_area_size = self.SCREEN_HEIGHT - 80
        padding = 40
        
        # Render Target Grid
        target_pos = (padding, (self.SCREEN_HEIGHT - grid_area_size) // 2 + 20)
        self.target_grid_rect = self._render_grid(self.screen, self.target_grid, target_pos, grid_area_size, "TARGET")

        # Render Playable Grid
        playable_pos = (self.SCREEN_WIDTH - grid_area_size - padding, target_pos[1])
        self.playable_grid_rect = self._render_grid(self.screen, self.playable_grid, playable_pos, grid_area_size, "YOUR GRID")

        # Render Push Highlight
        if self.last_push_info:
            highlight_surface = pygame.Surface((self.playable_grid_rect.width, self.playable_grid_rect.height), pygame.SRCALPHA)
            highlight_surface.fill((0,0,0,0))
            
            if self.last_push_info['type'] == 'row':
                y = self.last_push_info['index'] * self.grid_pixel_size
                rect = pygame.Rect(0, y, self.playable_grid_rect.width, self.grid_pixel_size)
            else: # col
                x = self.last_push_info['index'] * self.grid_pixel_size
                rect = pygame.Rect(x, 0, self.grid_pixel_size, self.playable_grid_rect.height)
            
            pygame.draw.rect(highlight_surface, self.COLOR_HIGHLIGHT + (100,), rect) # 100 alpha
            self.screen.blit(highlight_surface, self.playable_grid_rect.topleft)

        # Render Cursor
        cursor_x = self.playable_grid_rect.left + self.cursor_pos[0] * self.grid_pixel_size
        cursor_y = self.playable_grid_rect.top + self.cursor_pos[1] * self.grid_pixel_size
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.grid_pixel_size, self.grid_pixel_size)
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)

    def _render_ui(self):
        # Render Score
        score_text = f"Correct Pixels: {self.score} / {self.GRID_SIZE**2}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_pos = (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 20)
        self.screen.blit(score_surf, score_pos)

        # Render Moves
        moves_text = f"Moves Left: {self.moves_remaining}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        moves_pos = (self.SCREEN_WIDTH // 2 - moves_surf.get_width() // 2, 50)
        self.screen.blit(moves_surf, moves_pos)
        
        # Render Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))

            if np.array_equal(self.playable_grid, self.target_grid):
                win_text = "PERFECT!"
                win_surf = self.font_game_over.render(win_text, True, self.COLOR_HIGHLIGHT)
                win_pos = (self.SCREEN_WIDTH // 2 - win_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - win_surf.get_height() // 2)
                self.screen.blit(win_surf, win_pos)
            else:
                game_over_text = "GAME OVER"
                go_surf = self.font_game_over.render(game_over_text, True, self.COLOR_TEXT)
                go_pos = (self.SCREEN_WIDTH // 2 - go_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - go_surf.get_height() // 2)
                self.screen.blit(go_surf, go_pos)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Pixel Push")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = np.array([movement, space, shift])
        
        # For turn-based games, only step if an action is taken
        if movement != 0 or space != 0 or shift != 0 or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset.")
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In manual play, we need a delay to make it playable
        pygame.time.wait(100) 

    env.close()