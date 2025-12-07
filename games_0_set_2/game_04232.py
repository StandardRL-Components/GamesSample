
# Generated: 2025-08-28T01:48:30.568315
# Source Brief: brief_04232.md
# Brief Index: 4232

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through available colors. "
        "Press Space to place the selected color on the grid."
    )

    game_description = (
        "A strategic puzzle game. Recreate the target pixel art by filling the grid. "
        "Each color is a limited resource, so place your pixels wisely to achieve a "
        "95% match before you run out!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_COLORS = 10
        self.WIN_THRESHOLD = 95.0
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID_BG = (50, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.PALETTE = [
            (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
            (41, 173, 255), (131, 118, 156), (88, 88, 88), (200, 200, 200),
            (158, 0, 93), (0, 192, 137)
        ]
        self.BLANK_COLOR_INDEX = -1

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.end_message = ""
        self.target_image = None
        self.current_grid = None
        self.color_counts = None
        self.cursor_pos = None
        self.active_color_index = 0
        self.match_percentage = 0.0
        
        # This is a headless environment, but get_ticks is useful for animations
        self.internal_ticks = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.end_message = ""

        # Generate a new target image
        self.target_image = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))

        # Calculate initial color counts from the target image
        unique, counts = np.unique(self.target_image, return_counts=True)
        self.color_counts = np.zeros(self.NUM_COLORS, dtype=int)
        for i, color_idx in enumerate(unique):
            self.color_counts[color_idx] = counts[i]

        # Reset the player's grid
        self.current_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.BLANK_COLOR_INDEX, dtype=int)

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.active_color_index = 0
        if self.color_counts[self.active_color_index] == 0:
            self._cycle_color()

        self._recalculate_match()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.internal_ticks += 33 # Simulate 30fps for animations
        self.steps += 1
        reward = 0.0
        terminated = False
        
        movement, space_pressed, shift_pressed = action
        
        # --- Handle Actions ---
        if shift_pressed:
            self._cycle_color()
            # sfx: color_select

        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = [self.cursor_pos[0] % self.GRID_SIZE, self.cursor_pos[1] % self.GRID_SIZE]

        if space_pressed:
            reward, terminated = self._place_color()

        self.score += reward
        self._recalculate_match()

        # --- Check Termination Conditions ---
        if not terminated:
            if self.match_percentage >= self.WIN_THRESHOLD:
                terminated = True
                win_reward = 100
                self.score += win_reward
                reward += win_reward
                self.end_message = "YOU WIN!"
                # sfx: game_win
            elif np.all(self.current_grid != self.BLANK_COLOR_INDEX):
                terminated = True
                loss_reward = -100
                self.score += loss_reward
                reward += loss_reward
                self.end_message = "GRID FULL"
                # sfx: game_lose
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.end_message = "TIME UP"
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_color(self):
        cx, cy = self.cursor_pos
        reward = 0.0
        terminated = False

        if self.current_grid[cy, cx] == self.BLANK_COLOR_INDEX and self.color_counts[self.active_color_index] > 0:
            # sfx: place_pixel
            placed_color = self.active_color_index
            self.current_grid[cy, cx] = placed_color
            self.color_counts[placed_color] -= 1

            reward = 0.1 if self.current_grid[cy, cx] == self.target_image[cy, cx] else -0.02

            # Check for loss: ran out of a color that is still needed to complete the image
            if self.color_counts[placed_color] == 0:
                needed_mask = (self.target_image == placed_color)
                placed_mask = (self.current_grid == placed_color)
                if np.any(needed_mask & ~placed_mask):
                    terminated = True
                    loss_reward = -100
                    self.score += loss_reward
                    reward += loss_reward
                    self.end_message = "COLOR DEPLETED"
                    # sfx: game_lose

            # If the current color ran out (and not a loss), cycle to the next available one
            if self.color_counts[placed_color] == 0 and not terminated:
                self._cycle_color()
        
        return reward, terminated

    def _cycle_color(self):
        if np.sum(self.color_counts) == 0: return

        current_idx = self.active_color_index
        for _ in range(self.NUM_COLORS):
            current_idx = (current_idx + 1) % self.NUM_COLORS
            if self.color_counts[current_idx] > 0:
                self.active_color_index = current_idx
                return

    def _recalculate_match(self):
        correct_pixels = np.sum(self.current_grid == self.target_image)
        total_target_pixels = self.GRID_SIZE * self.GRID_SIZE
        self.match_percentage = (correct_pixels / total_target_pixels) * 100.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_target()
        self._render_grid()
        self._render_palette()
        self._render_ui()

        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_target(self):
        start_x, start_y, cell_size, padding = 20, 40, 10, 1
        title_surf = self.font_medium.render("Target Image", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (start_x, 10))

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.PALETTE[self.target_image[r, c]]
                rect = pygame.Rect(start_x + c * (cell_size + padding), start_y + r * (cell_size + padding), cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)

    def _render_grid(self):
        cell_size, padding = 28, 2
        grid_dim = self.GRID_SIZE * (cell_size + padding) - padding
        start_x = (self.WIDTH - grid_dim - 200) // 2 + 30
        start_y = (self.HEIGHT - grid_dim) // 2 + 10

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.current_grid[r, c]
                color = self.PALETTE[color_idx] if color_idx != self.BLANK_COLOR_INDEX else self.COLOR_GRID_BG
                rect = pygame.Rect(start_x + c * (cell_size + padding), start_y + r * (cell_size + padding), cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(start_x + cursor_x * (cell_size + padding) - padding, start_y + cursor_y * (cell_size + padding) - padding, cell_size + padding * 2, cell_size + padding * 2)
        
        blink_alpha = 128 + 127 * math.sin(self.internal_ticks * 0.005)
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, blink_alpha), cursor_surface.get_rect(), 3, border_radius=4)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_palette(self):
        start_x, start_y, swatch_size, padding = self.WIDTH - 150, 40, 30, 8
        title_surf = self.font_medium.render("Palette", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (start_x, 10))

        for i in range(self.NUM_COLORS):
            y_pos = start_y + i * (swatch_size + padding)
            rect = pygame.Rect(start_x, y_pos, swatch_size, swatch_size)
            
            color = self.PALETTE[i]
            if self.color_counts[i] == 0:
                color = tuple(c // 2 for c in color)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            if i == self.active_color_index and self.color_counts[i] > 0:
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, 3, border_radius=3)

            count_surf = self.font_small.render(f"x {self.color_counts[i]}", True, self.COLOR_TEXT)
            self.screen.blit(count_surf, (start_x + swatch_size + 10, y_pos + swatch_size // 4))

    def _render_ui(self):
        match_surf = self.font_medium.render(f"Match: {self.match_percentage:.1f}%", True, self.COLOR_TEXT)
        score_surf = self.font_medium.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(match_surf, (20, self.HEIGHT - 40))
        self.screen.blit(score_surf, (220, self.HEIGHT - 40))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        end_text_surf = self.font_large.render(self.end_message, True, self.COLOR_HIGHLIGHT)
        text_rect = end_text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        overlay.blit(end_text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "match_percentage": self.match_percentage,
            "colors_remaining_total": int(self.color_counts.sum()),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Matcher")
    clock = pygame.time.Clock()
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # --- Action mapping for human keyboard ---
        movement = 0 # 0=none
        space_pressed = 0 # 0=released
        shift_pressed = 0 # 0=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        # Since auto_advance is False, we only step on an action.
        # For human play, we need to decide when to step.
        # Let's step on any key press.
        action = [movement, space_pressed, shift_pressed]
        
        # We only want to step if an action is taken or for movement
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Match: {info['match_percentage']:.1f}%")

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if env.game_over:
            pygame.time.wait(2000) # Pause on win/loss screen
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Limit human play speed

    env.close()