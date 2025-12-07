
# Generated: 2025-08-27T21:16:12.337002
# Source Brief: brief_02732.md
# Brief Index: 2732

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the ninja on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Guide the ninja to collect numbers and reach a total score of 10 or more within 20 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_DIM = 10
        self.MAX_MOVES = 20
        self.TARGET_SCORE = 10
        self.INITIAL_NUMBERS = 30

        # --- Visuals ---
        self.COLOR_BG = (15, 15, 25) # Dark blue/black
        self.COLOR_GRID = (60, 60, 80)
        self.COLOR_NINJA = (0, 200, 255) # Bright Cyan
        self.COLOR_NINJA_OUTLINE = (255, 255, 255)
        self.COLOR_TEXT_UI = (220, 220, 240)
        self.COLOR_TEXT_SUCCESS = (100, 255, 100)
        self.COLOR_TEXT_FAIL = (255, 100, 100)
        self.NUMBER_COLORS = {
            1: (0, 255, 128),  # Green
            2: (255, 255, 0),  # Yellow
            3: (255, 128, 0)   # Orange
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_grid = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 60, bold=True)

        # --- Grid Layout Calculation ---
        self.UI_HEIGHT = 60
        self.GRID_AREA_SIZE = self.SCREEN_HEIGHT - self.UI_HEIGHT
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_DIM
        self.GRID_SIZE_PX = self.CELL_SIZE * self.GRID_DIM
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_SIZE_PX) // 2
        self.GRID_OFFSET_Y = self.UI_HEIGHT + (self.GRID_AREA_SIZE - self.GRID_SIZE_PX) // 2

        # --- Game State (initialized in reset) ---
        self.ninja_pos = None
        self.grid = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.steps = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.steps = 0

        # Place ninja randomly
        self.ninja_pos = self.np_random.integers(0, self.GRID_DIM, size=2, dtype=int)

        # Generate grid with numbers
        self.grid = np.zeros((self.GRID_DIM, self.GRID_DIM), dtype=int)
        empty_cells = list(np.argwhere(self.grid == 0))
        self.np_random.shuffle(empty_cells)
        
        num_to_place = min(self.INITIAL_NUMBERS, len(empty_cells))
        for r, c in empty_cells[:num_to_place]:
            self.grid[r, c] = self.np_random.integers(1, 4)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        terminated = False
        
        self.steps += 1

        # Store state before move for reward calculation
        old_pos = self.ninja_pos.copy()
        dist_before = self._get_distance_to_best_number(old_pos)

        # --- Apply Action ---
        if movement != 0: # If not a no-op
            self.moves_remaining -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            self.ninja_pos[0] = np.clip(self.ninja_pos[0] + dy, 0, self.GRID_DIM - 1)
            self.ninja_pos[1] = np.clip(self.ninja_pos[1] + dx, 0, self.GRID_DIM - 1)

        # --- Calculate Proximity Reward ---
        dist_after = self._get_distance_to_best_number(self.ninja_pos)
        if dist_after < dist_before:
            reward += 0.1 # Small reward for moving closer to a good number

        # --- Collect Number & Calculate Collection Reward ---
        r, c = self.ninja_pos
        if self.grid[r, c] > 0:
            collected_value = self.grid[r, c]
            self.score += collected_value
            self.grid[r, c] = 0
            reward += 1.0 # Reward for collecting any number
            # Sound effect placeholder:
            # play_sound("collect")

        # --- Check Termination Conditions ---
        if self.score >= self.TARGET_SCORE:
            reward += 100.0  # Big reward for winning
            terminated = True
            self.game_over = True
            # Sound effect placeholder:
            # play_sound("win")
        elif self.moves_remaining <= 0:
            reward -= 100.0  # Big penalty for losing
            terminated = True
            self.game_over = True
            # Sound effect placeholder:
            # play_sound("lose")
        
        # Max steps termination
        if self.steps >= 1000:
            terminated = True
            self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_distance_to_best_number(self, pos):
        """Calculates Manhattan distance to the nearest highest-value number."""
        for value in range(3, 0, -1):
            locations = np.argwhere(self.grid == value)
            if len(locations) > 0:
                distances = [abs(loc[0] - pos[0]) + abs(loc[1] - pos[1]) for loc in locations]
                return min(distances)
        return float('inf') # No numbers left

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "steps": self.steps,
            "ninja_pos": tuple(self.ninja_pos),
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE_PX)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE_PX, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw numbers
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                value = self.grid[r, c]
                if value > 0:
                    color = self.NUMBER_COLORS.get(value, (255, 255, 255))
                    text_surf = self.font_grid.render(str(value), True, color)
                    text_rect = text_surf.get_rect(center=(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    ))
                    self.screen.blit(text_surf, text_rect)

        # Draw ninja
        ninja_r, ninja_c = self.ninja_pos
        ninja_center_x = self.GRID_OFFSET_X + ninja_c * self.CELL_SIZE + self.CELL_SIZE // 2
        ninja_center_y = self.GRID_OFFSET_Y + ninja_r * self.CELL_SIZE + self.CELL_SIZE // 2
        
        # Using gfxdraw for antialiasing
        radius = self.CELL_SIZE // 3
        pygame.gfxdraw.filled_circle(self.screen, ninja_center_x, ninja_center_y, radius, self.COLOR_NINJA)
        pygame.gfxdraw.aacircle(self.screen, ninja_center_x, ninja_center_y, radius, self.COLOR_NINJA_OUTLINE)

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        moves_text = f"Moves: {self.moves_remaining}"
        target_text = f"Target: {self.TARGET_SCORE}"

        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT_UI)
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT_UI)
        target_surf = self.font_ui.render(target_text, True, self.COLOR_TEXT_UI)

        self.screen.blit(score_surf, (20, 20))
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH // 2 - moves_surf.get_width() // 2, 20))
        self.screen.blit(target_surf, (self.SCREEN_WIDTH - target_surf.get_width() - 20, 20))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.score >= self.TARGET_SCORE:
            text = "VICTORY"
            color = self.COLOR_TEXT_SUCCESS
        else:
            text = "GAME OVER"
            color = self.COLOR_TEXT_FAIL

        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Ninja Grid Collector")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False

        if not terminated and np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_remaining']}, Terminated: {terminated}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()