
# Generated: 2025-08-27T19:20:59.221714
# Source Brief: brief_02130.md
# Brief Index: 2130

        
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
        "Controls: Use arrow keys (↑↓←→) to draw a path from the green start to the red end point."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Connect the start and end points on the grid before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 25
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (255, 255, 255)
        self.COLOR_GRID = (220, 220, 220)
        self.COLOR_START = (50, 205, 50)  # LimeGreen
        self.COLOR_END = (255, 69, 0)      # OrangeRed
        self.COLOR_PATH = (65, 105, 225)   # RoyalBlue
        self.COLOR_PATH_HEAD = (255, 215, 0) # Gold
        self.COLOR_TEXT = (30, 30, 30)

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_gameover = pygame.font.Font(None, 60)

        # Initialize state variables
        self.start_pos = (0, 0)
        self.end_pos = (0, 0)
        self.path = []
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES

        # Generate a new puzzle
        while True:
            self.start_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            self.end_pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            dist = self._manhattan_distance(self.start_pos, self.end_pos)
            # Ensure start/end are not the same and are reasonably far apart
            if dist > self.GRID_SIZE // 2:
                break

        self.path = [self.start_pos]
        self.last_distance = self._manhattan_distance(self.start_pos, self.end_pos)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return the final state.
            return (self._get_observation(), 0, True, False, self._get_info())

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        # --- Handle No-Op ---
        if movement == 0:
            # No action taken, no state change, no reward.
            return (self._get_observation(), 0, False, False, self._get_info())

        # --- Calculate Next Position ---
        current_pos = self.path[-1]
        next_pos = list(current_pos)

        if movement == 1: # Up
            next_pos[1] -= 1
        elif movement == 2: # Down
            next_pos[1] += 1
        elif movement == 3: # Left
            next_pos[0] -= 1
        elif movement == 4: # Right
            next_pos[0] += 1
        next_pos = tuple(next_pos)

        # --- Validate Move ---
        is_off_grid = not (0 <= next_pos[0] < self.GRID_SIZE and 0 <= next_pos[1] < self.GRID_SIZE)
        is_self_crossing = next_pos in self.path

        if is_off_grid or is_self_crossing:
            # Invalid move, no state change, no reward.
            return (self._get_observation(), 0, False, False, self._get_info())

        # --- Process Valid Move ---
        self.steps += 1
        self.moves_remaining -= 1

        # Calculate distance-based reward
        new_distance = self._manhattan_distance(next_pos, self.end_pos)
        reward = self.last_distance - new_distance  # +1 if closer, -1 if further
        self.last_distance = new_distance

        self.path.append(next_pos)

        # --- Check for Termination ---
        terminated = False
        won = (next_pos == self.end_pos)
        lost = (self.moves_remaining <= 0 and not won)

        if won:
            # sound: win_sound.play()
            reward += 50  # Win bonus
            terminated = True
            self.game_over = True
        elif lost:
            # sound: lose_sound.play()
            reward -= 10  # Lose penalty
            terminated = True
            self.game_over = True

        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

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
            "moves_remaining": self.moves_remaining,
            "path_length": len(self.path),
        }

    def _render_game(self):
        # --- Render Grid ---
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # --- Render Start/End Points ---
        start_px = self._grid_to_pixel(self.start_pos)
        end_px = self._grid_to_pixel(self.end_pos)
        cell_rect = pygame.Rect(0, 0, self.CELL_SIZE, self.CELL_SIZE)

        cell_rect.center = start_px
        pygame.draw.rect(self.screen, self.COLOR_START, cell_rect)

        cell_rect.center = end_px
        pygame.draw.rect(self.screen, self.COLOR_END, cell_rect)

        # --- Render Path ---
        if len(self.path) > 1:
            path_pixels = [self._grid_to_pixel(p) for p in self.path]
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, path_pixels, 3)
            # Draw circles at joints for a smoother look
            for p_px in path_pixels:
                pygame.gfxdraw.filled_circle(self.screen, p_px[0], p_px[1], 5, self.COLOR_PATH)

        # --- Render Path Head ---
        if self.path:
            head_px = self._grid_to_pixel(self.path[-1])
            # Draw a glowing effect for the head
            pygame.gfxdraw.filled_circle(self.screen, head_px[0], head_px[1], 10, (*self.COLOR_PATH_HEAD, 60))
            pygame.gfxdraw.filled_circle(self.screen, head_px[0], head_px[1], 7, (*self.COLOR_PATH_HEAD, 120))
            pygame.gfxdraw.filled_circle(self.screen, head_px[0], head_px[1], 4, self.COLOR_PATH_HEAD)

    def _render_ui(self):
        # --- Render Moves Remaining ---
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 15))

        # --- Render Score ---
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

        # --- Render Game Over Message ---
        if self.game_over:
            won = self.path[-1] == self.end_pos
            message = "COMPLETE!" if won else "OUT OF MOVES"
            color = self.COLOR_START if won else self.COLOR_END

            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            self.screen.blit(overlay, (0, 0))

            gameover_text = self.font_gameover.render(message, True, color)
            text_rect = gameover_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(gameover_text, text_rect)

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates (e.g., (3, 4)) to pixel coordinates."""
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _manhattan_distance(self, p1, p2):
        """Calculates Manhattan distance between two grid points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs_initial = self._get_observation()
        assert test_obs_initial.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs_initial.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert info['moves_remaining'] == self.MAX_MOVES

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# Example of how to run the environment for manual play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()

    obs, info = env.reset()
    done = False

    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    running = True
    while running:
        action = [0, 0, 0]  # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if not env.game_over:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                
                if event.key == pygame.K_r:  # Reset key
                    obs, info = env.reset()
                    done = False
                
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

        # Draw the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        
        if done:
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()