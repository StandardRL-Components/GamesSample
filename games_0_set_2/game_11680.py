import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:21:19.054037
# Source Brief: brief_01680.md
# Brief Index: 1680
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A puzzle game where the player rotates rows and columns of a 3x3 grid
    of numbers to create matching lines of three.

    The action space is MultiDiscrete([5, 2, 2]):
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - up/down: Rotates a column
        - left/right: Rotates a row
    - actions[1]: Space button (0=released, 1=held)
    - actions[2]: Shift button (0=released, 1=held)
        - The combination of space and shift selects the row/column index (0, 1, or 2).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where you rotate rows and columns of a 3x3 grid to match three identical numbers in a line."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to rotate a row or column. "
        "Hold space to select the middle index (1) or shift for the last index (2). Default is the first index (0)."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 3
    CELL_SIZE = 100
    GRID_LINE_WIDTH = 4
    MAX_STEPS = 1000
    WIN_SCORE = 500
    ANIMATION_FRAMES = 10 # Lower is faster
    FLASH_FRAMES = 15 # Duration of flash effect

    # --- Colors ---
    COLOR_BG = pygame.Color("#1A1A2E")
    COLOR_GRID_BG = pygame.Color("#16213E")
    COLOR_GRID_LINES = pygame.Color("#0F3460")
    COLOR_SCORE = pygame.Color("#E94560")
    COLOR_INFO = pygame.Color("#FFFFFF")
    COLOR_SELECTOR_VALID = pygame.Color("#50C878")
    COLOR_FLASH = pygame.Color(255, 255, 255, 128) # Semi-transparent white
    NUMBER_COLORS = {
        1: pygame.Color("#FF4136"),  # Red
        2: pygame.Color("#2ECC40"),  # Green
        3: pygame.Color("#0074D9"),  # Blue
        4: pygame.Color("#FFDC00"),  # Yellow
        5: pygame.Color("#B10DC9"),  # Purple
        6: pygame.Color("#7FDBFF"),  # Aqua
        7: pygame.Color("#FF851B"),  # Orange
        8: pygame.Color("#F012BE"),  # Magenta
        9: pygame.Color("#3D9970"),  # Olive
    }

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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        self.grid_top_left = (
            (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2 + 20
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.game_state = 'INPUT' # INPUT, ROTATING, FLASHING, RESOLVING
        self.animation_progress = 0.0
        self.animation_info = None
        self.flash_timer = 0
        self.matched_cells = []

        self._initialize_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # --- State Machine Logic ---
        if self.game_state == 'INPUT':
            self._handle_input(action)
        elif self.game_state == 'ROTATING':
            self._update_rotation()
        elif self.game_state == 'FLASHING':
            self._update_flash()
        elif self.game_state == 'RESOLVING':
            reward += self._resolve_matches()

        # --- Check for termination ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100.0 # Goal-oriented reward
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Using terminated as it's a natural end condition

        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _initialize_grid(self):
        """Creates a grid and ensures it has no initial matches."""
        while True:
            self.grid = self.np_random.integers(1, 10, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._check_for_matches():
                break

    def _handle_input(self, action):
        """Processes player action if the game is in the INPUT state."""
        self.steps += 1
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 0:
            return # No-op

        # Decode index from space/shift
        if not space_held and not shift_held: index = 0
        elif space_held and not shift_held:  index = 1
        elif not space_held and shift_held:  index = 2
        else: return # Both held is a no-op

        # Decode rotation from movement
        if movement in [1, 2]: # Up/Down -> Column rotation
            rot_type = 'col'
            direction = 'up' if movement == 1 else 'down'
        elif movement in [3, 4]: # Left/Right -> Row rotation
            rot_type = 'row'
            direction = 'left' if movement == 3 else 'right'
        else:
            return

        # Start animation
        self.game_state = 'ROTATING'
        self.animation_progress = 0.0
        self.animation_info = {'type': rot_type, 'index': index, 'direction': direction}
        # Sound effect placeholder: // whoosh_sound.play()

    def _update_rotation(self):
        """Advances the rotation animation."""
        self.animation_progress += 1.0 / self.ANIMATION_FRAMES
        if self.animation_progress >= 1.0:
            self._apply_rotation_to_grid()
            self.animation_info = None
            self.game_state = 'RESOLVING' # Check for matches after rotation

    def _update_flash(self):
        """Advances the flash animation."""
        self.flash_timer -= 1
        if self.flash_timer <= 0:
            self.game_state = 'RESOLVING'

    def _resolve_matches(self):
        """Handles match scoring, number replacement, and checks for cascades."""
        score_from_matches = 0
        if self.matched_cells:
            # Sound effect placeholder: // resolve_sound.play()
            for r, c in self.matched_cells:
                score_from_matches += self.grid[r, c]
                self.grid[r, c] = self.np_random.integers(1, 10)
            self.score += score_from_matches
            self.matched_cells = []

        # Check for new (cascade) matches
        new_matches = self._check_for_matches()
        if new_matches:
            self.matched_cells = new_matches
            self.game_state = 'FLASHING'
            self.flash_timer = self.FLASH_FRAMES
            # Sound effect placeholder: // match_found_sound.play()
        else:
            # Board has settled, award potential reward and return to input
            self.game_state = 'INPUT'
            return score_from_matches + self._calculate_potential_reward()
        
        return score_from_matches

    def _apply_rotation_to_grid(self):
        """Permanently changes the self.grid data after an animation."""
        info = self.animation_info
        if info['type'] == 'row':
            row = self.grid[info['index'], :].copy()
            if info['direction'] == 'right':
                self.grid[info['index'], :] = np.roll(row, 1)
            else: # left
                self.grid[info['index'], :] = np.roll(row, -1)
        elif info['type'] == 'col':
            col = self.grid[:, info['index']].copy()
            if info['direction'] == 'down':
                self.grid[:, info['index']] = np.roll(col, 1)
            else: # up
                self.grid[:, info['index']] = np.roll(col, -1)

    def _check_for_matches(self):
        """Scans the grid for horizontal and vertical matches of three."""
        matches = set()
        # Check rows
        for r in range(self.GRID_SIZE):
            if self.grid[r, 0] == self.grid[r, 1] == self.grid[r, 2]:
                matches.update([(r, 0), (r, 1), (r, 2)])
        # Check columns
        for c in range(self.GRID_SIZE):
            if self.grid[0, c] == self.grid[1, c] == self.grid[2, c]:
                matches.update([(0, c), (1, c), (2, c)])
        return list(matches)

    def _calculate_potential_reward(self):
        """Calculates a small reward for creating pairs, encouraging setups."""
        potential = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check right neighbor
                if c < self.GRID_SIZE - 1 and self.grid[r, c] == self.grid[r, c + 1]:
                    potential += 0.1
                # Check bottom neighbor
                if r < self.GRID_SIZE - 1 and self.grid[r, c] == self.grid[r + 1, c]:
                    potential += 0.1
        return potential

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Draws the main game elements, including grid and numbers."""
        grid_rect = pygame.Rect(self.grid_top_left, (self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        if self.game_state == 'ROTATING':
            self._render_animated_rotation()
        else:
            self._render_static_grid()

        if self.game_state == 'FLASHING':
            self._render_flash_effect()
        
        # Draw grid lines on top
        for i in range(1, self.GRID_SIZE):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.grid_top_left[0] + i * self.CELL_SIZE, self.grid_top_left[1]),
                             (self.grid_top_left[0] + i * self.CELL_SIZE, self.grid_top_left[1] + grid_rect.height),
                             self.GRID_LINE_WIDTH)
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.grid_top_left[0], self.grid_top_left[1] + i * self.CELL_SIZE),
                             (self.grid_top_left[0] + grid_rect.width, self.grid_top_left[1] + i * self.CELL_SIZE),
                             self.GRID_LINE_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, grid_rect, self.GRID_LINE_WIDTH)


    def _render_static_grid(self):
        """Draws the grid when no animation is in progress."""
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._draw_number(self.grid[r, c], r, c)

    def _render_animated_rotation(self):
        """Draws the grid with one row/column in motion."""
        info = self.animation_info
        p = self.animation_progress
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                is_moving_cell = (info['type'] == 'row' and info['index'] == r) or \
                                 (info['type'] == 'col' and info['index'] == c)
                if not is_moving_cell:
                    self._draw_number(self.grid[r, c], r, c)

        # Draw the moving pieces
        if info['type'] == 'row':
            idx = info['index']
            direction_mult = 1 if info['direction'] == 'right' else -1
            for c in range(self.GRID_SIZE):
                offset_c = p * direction_mult
                wrapped_c = (c - direction_mult + self.GRID_SIZE) % self.GRID_SIZE
                self._draw_number(self.grid[idx, wrapped_c], idx, c, offset_c=offset_c)
        else: # col
            idx = info['index']
            direction_mult = 1 if info['direction'] == 'down' else -1
            for r in range(self.GRID_SIZE):
                offset_r = p * direction_mult
                wrapped_r = (r - direction_mult + self.GRID_SIZE) % self.GRID_SIZE
                self._draw_number(self.grid[wrapped_r, idx], r, idx, offset_r=offset_r)

    def _render_flash_effect(self):
        """Draws a flashing effect over matched cells."""
        if (self.flash_timer // 2) % 2 == 0: # Flicker effect
            for r, c in self.matched_cells:
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill(self.COLOR_FLASH)
                pos = (self.grid_top_left[0] + c * self.CELL_SIZE, self.grid_top_left[1] + r * self.CELL_SIZE)
                self.screen.blit(flash_surface, pos)

    def _draw_number(self, num, r, c, offset_r=0.0, offset_c=0.0):
        """Helper to draw a single number in a grid cell with potential offset."""
        color = self.NUMBER_COLORS.get(num, self.COLOR_INFO)
        text = self.font_large.render(str(num), True, color)
        text_rect = text.get_rect()
        
        cell_center_x = self.grid_top_left[0] + (c + 0.5) * self.CELL_SIZE + offset_c * self.CELL_SIZE
        cell_center_y = self.grid_top_left[1] + (r + 0.5) * self.CELL_SIZE + offset_r * self.CELL_SIZE
        
        text_rect.center = (int(cell_center_x), int(cell_center_y))
        self.screen.blit(text, text_rect)

    def _render_ui(self):
        """Renders the score and other UI text."""
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_INFO)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "LEVEL COMPLETE" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text_surf = self.font_large.render(win_text, True, self.COLOR_SCORE)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_state": self.game_state
        }

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run this, you'll need to unset the dummy video driver
    # and create a real display.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # We need a real display to run the manual test
    try:
        del os.environ['SDL_VIDEODRIVER']
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Grid Rotator")
    except pygame.error:
        print("Could not create display. Running in headless mode.")
        screen = None

    clock = pygame.time.Clock()

    # --- Manual Control Mapping ---
    # ARROWS: Movement action
    # SPACE/LSHIFT: Index selection
    #
    # Example: Press UP + SPACE to rotate column 1 upwards.
    #          Press RIGHT to rotate row 0 rightwards.
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        # Poll events to keep the window responsive and check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Only process new key presses when the game is ready for input
        if env.game_state == 'INPUT':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # Only step if an action is taken or the game is in an auto-advancing state
        if movement != 0 or env.game_state != 'INPUT':
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
        else: # If no action, just get the current observation for rendering
            obs = env._get_observation()


        # Render to the display if it exists
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()