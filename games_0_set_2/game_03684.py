
# Generated: 2025-08-28T00:05:43.094932
# Source Brief: brief_03684.md
# Brief Index: 3684

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    A puzzle game where the player rearranges pixels on a grid to match a target image.
    The player selects a pixel and can shift the entire row or column it belongs to.
    The goal is to solve the puzzle within the given time and move limits.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrows to shift the row/column of the selected pixel. "
        "Space/Shift to cycle through pixels."
    )

    # Short, user-facing description of the game
    game_description = (
        "Recreate the target image by strategically shifting colored pixels on a grid "
        "within the time and move limits."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 12, 10
        self.PIXEL_SIZE = 20
        self.GRID_BORDER = 4
        self.GLOW_RADIUS = int(self.PIXEL_SIZE * 0.8)
        self.MAX_MOVES = 150
        self.MAX_STEPS = 1800  # Corresponds to 60s at 30fps

        # --- Colors & Palette ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_DIM = (150, 150, 150)
        self.COLOR_GRID_BG = (40, 44, 52)
        self.COLOR_SUCCESS = (152, 251, 152)
        self.COLOR_FAIL = (255, 105, 97)
        self.PALETTE = [
            (0, 0, 0),          # 0: Black (for empty space in procedural gen)
            (255, 99, 71),      # 1: Tomato
            (255, 215, 0),      # 2: Gold
            (60, 179, 113),     # 3: Medium Sea Green
            (70, 130, 180),     # 4: Steel Blue
            (218, 112, 214),    # 5: Orchid
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_remaining = 0
        self.time_steps_remaining = 0
        self.target_grid = None
        self.playable_grid = None
        self.selector_pos = [0, 0]
        self.np_random = None
        self.correct_mask = None

        self.reset()
        
        # Self-check to ensure implementation correctness
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_remaining = self.MAX_MOVES
        self.time_steps_remaining = self.MAX_STEPS
        
        self._generate_puzzle()
        
        self.selector_pos = [0, 0]
        self.correct_mask = self.playable_grid == self.target_grid
        self.score = np.sum(self.correct_mask)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_steps_remaining -= 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        if movement != 0: # Movement takes precedence
            if self.moves_remaining > 0:
                self._perform_shift(movement)
                self.moves_remaining -= 1
                action_taken = True
        elif space_held:
            self._move_selector(1)
            action_taken = True
        elif shift_held:
            self._move_selector(-1)
            action_taken = True

        # --- Update State and Calculate Reward ---
        self.correct_mask = self.playable_grid == self.target_grid
        num_correct = np.sum(self.correct_mask)
        self.score = num_correct
        
        total_pixels = self.GRID_W * self.GRID_H
        reward = (num_correct * 1.0) - ((total_pixels - num_correct) * 0.1)

        # --- Check Termination Conditions ---
        is_solved = num_correct == total_pixels
        out_of_moves = self.moves_remaining <= 0 and action_taken and movement != 0
        out_of_time = self.time_steps_remaining <= 0
        
        terminated = is_solved or out_of_moves or out_of_time
        
        if terminated:
            self.game_over = True
            if is_solved:
                self.win = True
                reward += 100.0
            else:
                # Small penalty for failing
                reward -= 10.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_puzzle(self):
        """Creates a target image and a shuffled version for the player."""
        self.target_grid = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        
        # Procedurally draw a letter 'P'
        self.target_grid[2:8, 3] = 2
        self.target_grid[2:4, 4:6] = 2
        self.target_grid[4, 4:5] = 2

        self.playable_grid = self.target_grid.copy()
        
        # Scramble the puzzle with a few random moves
        num_scrambles = self.np_random.integers(15, 25)
        for _ in range(num_scrambles):
            axis = self.np_random.integers(0, 2) # 0 for row, 1 for col
            idx = self.np_random.integers(0, self.GRID_H if axis == 0 else self.GRID_W)
            direction = self.np_random.choice([-2, -1, 1, 2])
            if axis == 0:
                self.playable_grid[idx, :] = np.roll(self.playable_grid[idx, :], direction)
            else:
                self.playable_grid[:, idx] = np.roll(self.playable_grid[:, idx], direction)

    def _perform_shift(self, movement):
        """Shifts a row or column based on the movement action."""
        sel_x, sel_y = self.selector_pos
        # movement: 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            self.playable_grid[:, sel_x] = np.roll(self.playable_grid[:, sel_x], -1)
        elif movement == 2: # Down
            self.playable_grid[:, sel_x] = np.roll(self.playable_grid[:, sel_x], 1)
        elif movement == 3: # Left
            self.playable_grid[sel_y, :] = np.roll(self.playable_grid[sel_y, :], -1)
        elif movement == 4: # Right
            self.playable_grid[sel_y, :] = np.roll(self.playable_grid[sel_y, :], 1)

    def _move_selector(self, direction):
        """Moves the selector to the next/previous pixel in reading order."""
        sel_x, sel_y = self.selector_pos
        linear_index = sel_y * self.GRID_W + sel_x
        linear_index = (linear_index + direction) % (self.GRID_W * self.GRID_H)
        self.selector_pos = [linear_index % self.GRID_W, linear_index // self.GRID_W]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "time_steps_remaining": self.time_steps_remaining,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the target and playable grids."""
        grid_w_px = self.GRID_W * self.PIXEL_SIZE
        grid_h_px = self.GRID_H * self.PIXEL_SIZE
        
        # --- Target Grid ---
        target_x = 30
        target_y = (self.HEIGHT - grid_h_px) // 2
        self._draw_text("TARGET", (target_x + grid_w_px // 2, target_y - 20), self.font_m, self.COLOR_TEXT_DIM)
        self._render_grid(self.target_grid, (target_x, target_y), self.PALETTE, self.PIXEL_SIZE)

        # --- Playable Grid ---
        playable_x = self.WIDTH - grid_w_px - 30
        playable_y = (self.HEIGHT - grid_h_px) // 2
        self._draw_text("WORKSPACE", (playable_x + grid_w_px // 2, playable_y - 20), self.font_m, self.COLOR_TEXT)
        self._render_grid(self.playable_grid, (playable_x, playable_y), self.PALETTE, self.PIXEL_SIZE, self.correct_mask)
        
        # --- Render Selector ---
        sel_x, sel_y = self.selector_pos
        sel_px = playable_x + sel_x * self.PIXEL_SIZE
        sel_py = playable_y + sel_y * self.PIXEL_SIZE
        
        # Pulsing effect for selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        color = (255, 255, 255)
        
        # Draw a thicker, pulsing rectangle
        thickness = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, color, (sel_px, sel_py, self.PIXEL_SIZE, self.PIXEL_SIZE), thickness)

    def _render_grid(self, grid, pos, palette, pixel_size, glow_mask=None):
        """Helper function to draw a grid of pixels."""
        start_x, start_y = pos
        grid_h, grid_w = grid.shape

        # Draw grid background
        bg_rect = (start_x - self.GRID_BORDER, start_y - self.GRID_BORDER,
                   grid_w * pixel_size + 2 * self.GRID_BORDER, grid_h * pixel_size + 2 * self.GRID_BORDER)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, bg_rect, border_radius=5)
        
        for y in range(grid_h):
            for x in range(grid_w):
                color_index = grid[y, x]
                if color_index == 0: continue # Skip drawing empty pixels
                
                color = palette[color_index]
                px, py = start_x + x * pixel_size, start_y + y * pixel_size
                
                # Draw glow for correctly placed pixels
                if glow_mask is not None and glow_mask[y, x]:
                    glow_color = color
                    # Use gfxdraw for smooth, transparent circles
                    # pygame.gfxdraw.filled_circle takes (surface, x, y, radius, color)
                    # color must be a tuple with alpha (r, g, b, a)
                    center_x, center_y = px + pixel_size // 2, py + pixel_size // 2
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.GLOW_RADIUS, (*glow_color, 40))
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.GLOW_RADIUS * 0.7), (*glow_color, 60))
                
                # Draw the main pixel
                pygame.draw.rect(self.screen, color, (px, py, pixel_size, pixel_size))

    def _render_ui(self):
        """Renders the UI text for score, moves, and time."""
        # --- Info Panel ---
        self._draw_text(f"Correct: {self.score}/{self.GRID_W * self.GRID_H}", (self.WIDTH // 2, 30), self.font_m, self.COLOR_TEXT)
        self._draw_text(f"Moves: {self.moves_remaining}", (self.WIDTH // 2, 60), self.font_m, self.COLOR_TEXT)
        
        time_sec = max(0, self.time_steps_remaining // 30)
        self._draw_text(f"Time: {time_sec}", (self.WIDTH // 2, 90), self.font_m, self.COLOR_TEXT)

        # --- Game Over/Success Overlay ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                self._draw_text("PUZZLE SOLVED!", (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_l, self.COLOR_SUCCESS)
                final_score_text = f"Final Score: {int(self.score)}"
                self._draw_text(final_score_text, (self.WIDTH // 2, self.HEIGHT // 2 + 40), self.font_m, self.COLOR_TEXT)
            else:
                self._draw_text("GAME OVER", (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_l, self.COLOR_FAIL)
                reason = "Out of Time" if self.time_steps_remaining <= 0 else "Out of Moves"
                self._draw_text(reason, (self.WIDTH // 2, self.HEIGHT // 2 + 40), self.font_m, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color):
        """Helper to draw centered text."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy', or 'windows' as appropriate

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This setup allows a human to play the game.
    pygame.display.set_caption("Pixel Shifter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # --- Action Mapping for Manual Play ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # In a turn-based game, we only want to step when a key is pressed.
        # This loop waits for a keydown event.
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        if terminated: break
        
        # Only step if a key was pressed to match the turn-based nature
        if event_happened:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for smooth input handling

    print("Game Over!")
    env.close()