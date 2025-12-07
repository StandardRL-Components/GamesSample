
# Generated: 2025-08-27T23:48:52.776917
# Source Brief: brief_03592.md
# Brief Index: 3592

        
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
        "Controls: Use arrow keys to push all pixels Up, Down, Left, or Right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push pixels to fill the grid! Reach 90% fill before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_W = self.WIDTH // self.CELL_SIZE
        self.GRID_H = self.HEIGHT // self.CELL_SIZE
        self.total_cells = self.GRID_W * self.GRID_H

        # Game Parameters
        self.MAX_STEPS = 3000
        self.WIN_PERCENTAGE = 0.90
        self.INITIAL_PIXEL_COUNT = int(self.total_cells * 0.05)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 20, bold=True)
        self.font_title = pygame.font.SysFont('monospace', 36, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_BAR_BG = (50, 60, 80)
        self.COLOR_BAR_FILL = (100, 200, 255)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        self.PIXEL_COLORS = [
            (255, 0, 128), (0, 255, 255), (255, 255, 0),
            (0, 255, 128), (255, 128, 0), (128, 0, 255)
        ]
        
        # State variables (initialized in reset)
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pixel_count = 0
        self.action_feedback_timer = 0

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_feedback_timer = 0

        # Create the grid and populate it
        self.grid = [[None for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        
        # Generate unique random positions for initial pixels
        all_positions = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        if len(all_positions) > 0:
            initial_indices = self.np_random.choice(len(all_positions), self.INITIAL_PIXEL_COUNT, replace=False)
            
            for index in initial_indices:
                x, y = all_positions[index]
                color_index = self.np_random.integers(0, len(self.PIXEL_COLORS))
                self.grid[y][x] = self.PIXEL_COLORS[color_index]

        self.pixel_count = self._count_pixels()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info(),
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        
        if movement != 0:
            # Sound effect placeholder
            # play_sound('push')
            self._push_pixels(movement)
            self.action_feedback_timer = 3 # frames to show feedback

        self.pixel_count = self._count_pixels()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += term_reward

        if terminated:
            self.game_over = True
            # Sound effect placeholder
            # if term_reward > 0: play_sound('win') else: play_sound('lose')

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _push_pixels(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(direction, (0, 0))
        
        new_grid = [[None for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y][x] is not None:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                        new_grid[ny][nx] = self.grid[y][x]
        
        self.grid = new_grid

    def _count_pixels(self):
        count = 0
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    count += 1
        return count

    def _calculate_reward(self):
        fill_reward = self.pixel_count * 0.1
        empty_penalty = (self.total_cells - self.pixel_count) * -0.02
        return fill_reward + empty_penalty

    def _check_termination(self):
        fill_percentage = self.pixel_count / self.total_cells if self.total_cells > 0 else 0
        
        if fill_percentage >= self.WIN_PERCENTAGE:
            return True, 100.0
        
        if self.steps >= self.MAX_STEPS:
            return True, -100.0
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw action feedback (border flash)
        if self.action_feedback_timer > 0:
            self.action_feedback_timer -= 1
            flash_color = self.PIXEL_COLORS[self.np_random.integers(0, len(self.PIXEL_COLORS))]
            pygame.draw.rect(self.screen, flash_color, (0, 0, self.WIDTH, self.HEIGHT), 4)

        # Draw pixels
        for y, row in enumerate(self.grid):
            for x, color in enumerate(row):
                if color is not None:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

    def _render_ui(self):
        fill_percentage = self.pixel_count / self.total_cells if self.total_cells > 0 else 0
        bar_y, bar_height, bar_width = 5, 20, self.WIDTH - 20
        
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, bar_y, bar_width, bar_height), border_radius=5)
        fill_width = max(0, bar_width * fill_percentage)
        pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (10, bar_y, fill_width, bar_height), border_radius=5)
        
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (15, bar_y + bar_height + 5))
        
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, bar_y + bar_height + 5))
        self.screen.blit(score_text, score_rect)

        fill_text_str = f"{fill_percentage:.1%}"
        fill_text = self.font_ui.render(fill_text_str, True, self.COLOR_TEXT)
        fill_text_rect = fill_text.get_rect(center=(self.WIDTH / 2, bar_y + bar_height / 2))
        self.screen.blit(fill_text, fill_text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if fill_percentage >= self.WIN_PERCENTAGE:
                end_text_str, end_color = "GRID FILLED!", self.COLOR_WIN
            else:
                end_text_str, end_color = "TIME UP!", self.COLOR_LOSE

            end_text = self.font_title.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fill_percentage": self.pixel_count / self.total_cells if self.total_cells > 0 else 0,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pusher")
    
    action = env.action_space.sample()
    action[0] = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_over = False
                    action[0] = 0
                
                if not game_over:
                    key_map = {
                        pygame.K_UP: 1, pygame.K_DOWN: 2,
                        pygame.K_LEFT: 3, pygame.K_RIGHT: 4
                    }
                    action[0] = key_map.get(event.key, 0)

                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        game_over = terminated
                        action[0] = 0
        
        frame = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()