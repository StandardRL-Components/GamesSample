
# Generated: 2025-08-27T23:04:46.731804
# Source Brief: brief_03340.md
# Brief Index: 3340

        
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
    """
    An environment where the player manipulates falling pixel blocks to fill a grid.

    The player controls a 'pusher' at the bottom of the screen. Each turn, the
    player can move the pusher left or right. A block then falls from the top.
    If the block lands on the pusher, it is deflected to an adjacent empty space.
    The goal is to fill the entire grid within a limited number of moves.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: 3 for left, 4 for right. Others are no-op.
    - `action[1]`: Unused.
    - `action[2]`: Unused.

    **Observation Space:** A 640x400 RGB image of the game state.

    **Rewards:**
    - +0.1 for each block successfully placed in an empty grid cell.
    - +100 for completely filling the grid (winning).
    - -100 for running out of moves before filling the grid (losing).

    **Termination:**
    - The episode ends when the grid is full (win).
    - The episode ends when the move limit (50) is reached (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the pusher. Fill the grid with falling blocks before you run out of moves."
    )

    game_description = (
        "A puzzle game of spatial reasoning. Manipulate falling pixel blocks to completely fill a grid within a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 20
        self.CELL_SIZE = 18
        self.MAX_MOVES = 50

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 65)
        self.COLOR_PUSHER = (230, 230, 255)
        self.COLOR_PUSHER_OUTLINE = (180, 180, 200)
        self.BLOCK_COLORS = [
            (0, 0, 0),  # Index 0 is empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER_BG = (0, 0, 0, 180)

        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Grid positioning ---
        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_left = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_top = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.pusher_pos_x = None
        self.current_block = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.win_status = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.pusher_pos_x = self.GRID_WIDTH // 2
        self.moves_left = self.MAX_MOVES
        self.score = 0.0
        self.game_over = False
        self.win_status = False

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]

        # --- Apply Action: Move Pusher ---
        if movement == 3:  # Left
            self.pusher_pos_x = max(0, self.pusher_pos_x - 1)
        elif movement == 4:  # Right
            self.pusher_pos_x = min(self.GRID_WIDTH - 1, self.pusher_pos_x + 1)

        self.moves_left -= 1
        cells_before = np.count_nonzero(self.grid)

        # --- Simulate Block Drop ---
        self._simulate_block_drop()

        cells_after = np.count_nonzero(self.grid)
        placed_blocks = cells_after - cells_before
        
        # --- Calculate Reward ---
        reward = float(placed_blocks * 0.1)

        # --- Check Termination ---
        self.win_status = np.all(self.grid > 0)
        loss_status = self.moves_left <= 0
        terminated = self.win_status or loss_status
        
        if terminated:
            self.game_over = True
            if self.win_status:
                reward += 100.0
            elif loss_status: # Only penalize if it's a loss, not a win on the last move
                reward -= 100.0

        self.score += reward

        # --- Spawn next block if not game over ---
        if not self.game_over:
            self._spawn_new_block()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _simulate_block_drop(self):
        """Simulates a single block falling until it lands."""
        block_x = self.current_block['x']
        block_y = self.current_block['y']
        color_idx = self.current_block['color_idx']

        final_y = 0
        for y_check in range(self.GRID_HEIGHT):
            if self.grid[y_check, block_x] != 0:
                final_y = y_check - 1
                break
            final_y = y_check
        
        final_x = block_x

        # Deflection logic
        if final_y == self.GRID_HEIGHT - 1 and final_x == self.pusher_pos_x:
            # Try to deflect left
            if final_x > 0 and self.grid[final_y, final_x - 1] == 0:
                final_x -= 1
            # Else, try to deflect right
            elif final_x < self.GRID_WIDTH - 1 and self.grid[final_y, final_x + 1] == 0:
                final_x += 1
        
        if 0 <= final_y < self.GRID_HEIGHT and 0 <= final_x < self.GRID_WIDTH:
            self.grid[final_y, final_x] = color_idx
            # sfx: block_land.wav

    def _spawn_new_block(self):
        """Creates a new block at a random position at the top of the grid."""
        x = self.np_random.integers(0, self.GRID_WIDTH)
        color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS))
        self.current_block = {'x': x, 'y': 0, 'color_idx': color_idx}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "grid_filled_percent": np.count_nonzero(self.grid) / self.grid.size,
        }

    def _render_game(self):
        """Renders the grid, blocks, and pusher."""
        # Draw grid background
        pygame.draw.rect(
            self.screen,
            self.COLOR_GRID,
            (self.grid_left, self.grid_top, self.grid_pixel_width, self.grid_pixel_height),
        )

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    self._draw_block(c, r, self.BLOCK_COLORS[color_idx])

        # Draw current falling block
        if not self.game_over:
            self._draw_block(
                self.current_block['x'],
                self.current_block['y'],
                self.BLOCK_COLORS[self.current_block['color_idx']],
                is_falling=True
            )

        # Draw pusher
        pusher_x = self.grid_left + self.pusher_pos_x * self.CELL_SIZE
        pusher_y = self.grid_top + self.grid_pixel_height + 5
        pusher_rect_outline = pygame.Rect(pusher_x, pusher_y, self.CELL_SIZE, self.CELL_SIZE // 2)
        pusher_rect = pusher_rect_outline.inflate(-4, -4)
        
        pygame.draw.rect(self.screen, self.COLOR_PUSHER_OUTLINE, pusher_rect_outline, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PUSHER, pusher_rect, border_radius=2)
        
    def _draw_block(self, c, r, color, is_falling=False):
        """Draws a single block with a 3D effect."""
        x = self.grid_left + c * self.CELL_SIZE
        y = self.grid_top + r * self.CELL_SIZE
        
        # Create a slightly darker color for the shadow/outline
        shadow_color = tuple(max(0, val - 40) for val in color)
        
        rect_outline = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        rect_main = rect_outline.inflate(-4, -4)

        pygame.draw.rect(self.screen, shadow_color, rect_outline, border_radius=3)
        pygame.draw.rect(self.screen, color, rect_main, border_radius=2)
        
        if is_falling:
            # Add a small indicator at the top of the column
            indicator_x = self.grid_left + c * self.CELL_SIZE + self.CELL_SIZE // 2
            indicator_y = self.grid_top - 8
            pygame.draw.circle(self.screen, color, (indicator_x, indicator_y), 4)


    def _render_ui(self):
        """Renders UI text like score and moves left."""
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 15))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_BG)
            self.screen.blit(overlay, (0, 0))

            if self.win_status:
                end_text_str = "YOU WIN!"
            else:
                end_text_str = "GAME OVER"
                
            end_text = self.font_gameover.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# To run and play the game manually
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Filler")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        move = 0
        if keys[pygame.K_LEFT]:
            move = 3
        elif keys[pygame.K_RIGHT]:
            move = 4
        
        # Since auto_advance is False, we only step on a key press
        if move != 0:
            action[0] = move
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("--- GAME OVER --- Press 'R' to reset.")
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()