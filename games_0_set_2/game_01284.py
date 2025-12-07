
# Generated: 2025-08-27T16:39:18.727431
# Source Brief: brief_01284.md
# Brief Index: 1284

        
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
        "Controls: Use arrow keys to push the selected pixel's row/column. "
        "Press Space to cycle the selection to the next pixel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A pixel-pushing puzzle. Rearrange the pixels on your grid to match the "
        "target image. You only have a limited number of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_DIM = 10
    PIXEL_SIZE = 25
    GRID_BORDER = 3
    MAX_MOVES = 20
    SCRAMBLE_DEPTH = 5 # How many random moves to scramble the puzzle

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GRID_BG = (40, 50, 70)
    COLOR_HIGHLIGHT = (255, 255, 100)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

    # Palette for the puzzle pixels
    PALETTE = [
        (230, 69, 57),   # Red
        (247, 182, 58),  # Orange
        (247, 243, 112), # Yellow
        (112, 214, 112), # Green
        (58, 182, 247),  # Blue
        (112, 112, 214), # Indigo
        (182, 58, 247),  # Violet
        (255, 255, 255)  # White
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid_size_px = self.GRID_DIM * self.PIXEL_SIZE
        self.target_grid_pos = (
            self.SCREEN_WIDTH // 4 - self.grid_size_px // 2,
            (self.SCREEN_HEIGHT - self.grid_size_px) // 2 + 20
        )
        self.player_grid_pos = (
            self.SCREEN_WIDTH * 3 // 4 - self.grid_size_px // 2,
            (self.SCREEN_HEIGHT - self.grid_size_px) // 2 + 20
        )

        self.current_grid = None
        self.target_grid = None
        self.selected_pixel = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_state = None
        self.rng = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None
        self.moves_left = self.MAX_MOVES
        self.selected_pixel = (self.GRID_DIM // 2, self.GRID_DIM // 2)

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        action_taken = False
        
        old_correct_pixels = self._count_correct_pixels()

        if movement > 0: # A push action is a "move"
            self._push_pixel(movement)
            self.moves_left -= 1
            action_taken = True
        elif space_press: # A selection change is not a "move"
            self._cycle_selection()
            
        self.steps += 1
        
        if action_taken:
            new_correct_pixels = self._count_correct_pixels()
            # Reward for improving the board state + cost of move
            reward += (new_correct_pixels - old_correct_pixels)
            reward -= 0.1

        is_win = np.array_equal(self.current_grid, self.target_grid)
        is_loss = self.moves_left <= 0 and not is_win
        
        terminated = False
        if is_win:
            reward += 100
            self.game_over = True
            self.win_state = "WIN"
            terminated = True
        elif is_loss:
            reward += -10
            self.game_over = True
            self.win_state = "LOSE"
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_puzzle(self):
        # Generate a random target image
        self.target_grid = self.rng.integers(0, len(self.PALETTE), size=(self.GRID_DIM, self.GRID_DIM))
        self.current_grid = self.target_grid.copy()

        # Scramble the puzzle by applying random moves
        for _ in range(self.SCRAMBLE_DEPTH):
            # Temporarily set selected pixel for scrambling
            scramble_pos = (self.rng.integers(0, self.GRID_DIM), self.rng.integers(0, self.GRID_DIM))
            scramble_dir = self.rng.integers(1, 5) # 1-4 for up/down/left/right
            self._push_pixel(scramble_dir, grid=self.current_grid, pos=scramble_pos)

    def _push_pixel(self, direction, grid=None, pos=None):
        # sound: 'pixel_shift.wav'
        if grid is None:
            grid = self.current_grid
        if pos is None:
            pos = self.selected_pixel
        
        px, py = pos

        if direction == 1: # Up
            col = grid[:, px].copy()
            grid[:, px] = np.roll(col, -1)
        elif direction == 2: # Down
            col = grid[:, px].copy()
            grid[:, px] = np.roll(col, 1)
        elif direction == 3: # Left
            row = grid[py, :].copy()
            grid[py, :] = np.roll(row, -1)
        elif direction == 4: # Right
            row = grid[py, :].copy()
            grid[py, :] = np.roll(row, 1)

    def _cycle_selection(self):
        # sound: 'select_tick.wav'
        x, y = self.selected_pixel
        x += 1
        if x >= self.GRID_DIM:
            x = 0
            y += 1
            if y >= self.GRID_DIM:
                y = 0
        self.selected_pixel = (x, y)

    def _count_correct_pixels(self):
        return np.sum(self.current_grid == self.target_grid)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "correct_pixels": self._count_correct_pixels(),
        }

    def _render_game(self):
        # Draw Target Grid
        self._draw_grid(self.target_grid, self.target_grid_pos)
        # Draw Player Grid
        self._draw_grid(self.current_grid, self.player_grid_pos)
        # Draw Selection Highlight
        self._draw_selection()

    def _draw_grid(self, grid_data, top_left_pos):
        # Draw grid background
        bg_rect = pygame.Rect(
            top_left_pos[0] - self.GRID_BORDER,
            top_left_pos[1] - self.GRID_BORDER,
            self.grid_size_px + self.GRID_BORDER * 2,
            self.grid_size_px + self.GRID_BORDER * 2
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, bg_rect, border_radius=5)

        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_index = grid_data[y, x]
                color = self.PALETTE[color_index]
                rect = pygame.Rect(
                    top_left_pos[0] + x * self.PIXEL_SIZE,
                    top_left_pos[1] + y * self.PIXEL_SIZE,
                    self.PIXEL_SIZE,
                    self.PIXEL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

    def _draw_selection(self):
        if self.game_over:
            return
            
        sel_x, sel_y = self.selected_pixel
        rect = pygame.Rect(
            self.player_grid_pos[0] + sel_x * self.PIXEL_SIZE,
            self.player_grid_pos[1] + sel_y * self.PIXEL_SIZE,
            self.PIXEL_SIZE,
            self.PIXEL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, width=3)
        
        # Add a subtle glow
        glow_rect = rect.inflate(4, 4)
        pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, glow_rect, width=1, border_radius=2)

    def _render_ui(self):
        # Draw Titles
        target_text = self.font_small.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.target_grid_pos[0], self.target_grid_pos[1] - 25))

        player_text = self.font_small.render("YOUR GRID", True, self.COLOR_TEXT)
        self.screen.blit(player_text, (self.player_grid_pos[0], self.player_grid_pos[1] - 25))

        # Draw Moves Left
        moves_text = self.font_main.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        text_rect = moves_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(moves_text, text_rect)
        
        # Draw Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)

        # Draw Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_state == "WIN":
                # sound: 'win_jingle.wav'
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else: # LOSE
                # sound: 'lose_sound.wav'
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_LOSE)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Set to 'human' for a playable window
    render_mode = "human" # "human" or "rgb_array"
    
    if render_mode == "human":
        pygame.display.set_caption("Pixel Pusher")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    running = True
    terminated = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and not terminated:
                    action = [0, 0, 0] # no-op, release, release
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                    elif event.key == pygame.K_r: # Reset key
                         obs, info = env.reset()
                         terminated = False
                         print("--- Game Reset ---")
                         continue

                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            
            # Draw the environment's screen to the display
            game_surface = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
            screen.blit(game_surface, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30)
        else: # rgb_array mode for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("--- Episode Finished ---")
                env.reset()

    env.close()