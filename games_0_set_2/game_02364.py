
# Generated: 2025-08-27T20:08:57.436613
# Source Brief: brief_02364.md
# Brief Index: 2364

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
from fractions import Fraction
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your blue square around the grid. "
        "Collect fractions that are equivalent to the target fraction shown in the top-right."
    )

    game_description = (
        "An educational puzzle game. Navigate a grid to collect correct fractions and "
        "achieve mathematical mastery, while avoiding the incorrect ones. "
        "The target fraction changes as you progress."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.font_ui = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_fraction = pygame.font.SysFont('Arial', 18, bold=True)
        self.font_game_over = pygame.font.SysFont('Arial', 60, bold=True)

        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (50, 50, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_CORRECT = (50, 255, 150)
        self.COLOR_INCORRECT = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_WIN = (200, 255, 200)
        self.COLOR_LOSE = (255, 200, 200)
        
        # --- Game Grid Configuration ---
        self.grid_size = (12, 8)  # (cols, rows)
        self.cell_size = 45
        self.grid_width = self.grid_size[0] * self.cell_size
        self.grid_height = self.grid_size[1] * self.cell_size
        self.grid_offset_x = (self.screen_width - self.grid_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_height) // 2

        # --- Game State Variables ---
        self.rng = None
        self.player_pos = None
        self.grid = None
        self.target_fraction = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.correct_collected = 0
        self.incorrect_collected = 0
        
        # --- Game Parameters ---
        self.max_steps = 1000
        self.win_condition_correct = 15
        self.lose_condition_incorrect = 5
        self.num_correct_fractions = 5
        self.num_incorrect_fractions = 10

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.correct_collected = 0
        self.incorrect_collected = 0
        
        self.player_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        
        self._generate_new_target_fraction()
        self._populate_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        dist_before = self._find_nearest_correct_dist()

        # --- Update Player Position ---
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        # Clamp position to grid bounds
        px = max(0, min(self.grid_size[0] - 1, px))
        py = max(0, min(self.grid_size[1] - 1, py))
        self.player_pos = [px, py]

        dist_after = self._find_nearest_correct_dist()

        # --- Movement Reward ---
        if dist_before > 0: # Only apply if a target exists
            if dist_after < dist_before:
                reward += 1.0  # Moved closer to a correct fraction
            elif dist_after > dist_before:
                reward -= 0.1 # Moved further away

        # --- Fraction Collection ---
        cell = self.grid[px][py]
        if cell:
            if cell["type"] == "correct":
                # sfx: correct_collect.wav
                reward += 10.0
                self.score += 10
                self.correct_collected += 1
                if self.correct_collected % 5 == 0 and self.correct_collected < self.win_condition_correct:
                    self._generate_new_target_fraction()
                    self._populate_grid()
            else: # incorrect
                # sfx: incorrect_collect.wav
                reward -= 20.0
                self.score -= 20
                self.incorrect_collected += 1
            
            self.grid[px][py] = None # Remove fraction from grid

        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.correct_collected >= self.win_condition_correct:
            # sfx: win_game.wav
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
            self.score += 100
        elif self.incorrect_collected >= self.lose_condition_incorrect:
            # sfx: lose_game.wav
            terminated = True
            self.game_over = True
            reward -= 100.0
            self.score -= 100
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_new_target_fraction(self):
        level = self.correct_collected // 5
        if level == 0:
            self.target_fraction = Fraction(1, 2)
        elif level == 1:
            self.target_fraction = Fraction(1, 3)
        else: # level 2+
            den = self.rng.randint(4, 10)
            num = self.rng.randint(1, den - 1)
            self.target_fraction = Fraction(num, den)

    def _populate_grid(self):
        self.grid = [[None for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        empty_cells = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) != tuple(self.player_pos):
                    empty_cells.append((x, y))
        
        self.rng.shuffle(empty_cells)

        # Place correct fractions
        for _ in range(self.num_correct_fractions):
            if not empty_cells: break
            pos = empty_cells.pop()
            multiplier = self.rng.randint(2, 4)
            f = Fraction(self.target_fraction.numerator * multiplier, self.target_fraction.denominator * multiplier)
            self.grid[pos[0]][pos[1]] = {"fraction": f, "type": "correct"}

        # Place incorrect fractions
        for _ in range(self.num_incorrect_fractions):
            if not empty_cells: break
            pos = empty_cells.pop()
            while True:
                den = self.rng.randint(self.target_fraction.denominator, 12)
                num = self.rng.randint(1, den)
                if den == 0: continue
                f = Fraction(num, den)
                if f != self.target_fraction:
                    self.grid[pos[0]][pos[1]] = {"fraction": f, "type": "incorrect"}
                    break
    
    def _find_nearest_correct_dist(self):
        min_dist = float('inf')
        found = False
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                cell = self.grid[x][y]
                if cell and cell["type"] == "correct":
                    dist = abs(self.player_pos[0] - x) + abs(self.player_pos[1] - y)
                    min_dist = min(min_dist, dist)
                    found = True
        return min_dist if found else 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.grid_size[0] + 1):
            start_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.grid_size[1] + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + y * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            
        # Draw fractions
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                cell = self.grid[x][y]
                if cell:
                    f = cell["fraction"]
                    text = f"{f.numerator}/{f.denominator}"
                    color = self.COLOR_CORRECT if cell["type"] == "correct" else self.COLOR_INCORRECT
                    
                    text_surf = self.font_fraction.render(text, True, color)
                    text_rect = text_surf.get_rect(center=(
                        self.grid_offset_x + x * self.cell_size + self.cell_size // 2,
                        self.grid_offset_y + y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text_surf, text_rect)

        # Draw player
        player_rect = pygame.Rect(
            self.grid_offset_x + self.player_pos[0] * self.cell_size,
            self.grid_offset_y + self.player_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-8, -8), border_radius=4)
        
    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Target Fraction
        target_text = f"Target: {self.target_fraction.numerator}/{self.target_fraction.denominator}"
        target_surf = self.font_ui.render(target_text, True, self.COLOR_UI_TEXT)
        target_rect = target_surf.get_rect(topright=(self.screen_width - 15, 10))
        self.screen.blit(target_surf, target_rect)
        
        # Progress bars
        correct_bar_w = (self.correct_collected / self.win_condition_correct) * 150
        incorrect_bar_w = (self.incorrect_collected / self.lose_condition_incorrect) * 150
        pygame.draw.rect(self.screen, self.COLOR_GRID, (15, 35, 150, 10))
        pygame.draw.rect(self.screen, self.COLOR_CORRECT, (15, 35, correct_bar_w, 10))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.screen_width - 165, 35, 150, 10))
        pygame.draw.rect(self.screen, self.COLOR_INCORRECT, (self.screen_width - 165, 35, incorrect_bar_w, 10))
        
        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win:
                text = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                text = "GAME OVER"
                color = self.COLOR_LOSE
                
            text_surf = self.font_game_over.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "correct_collected": self.correct_collected,
            "incorrect_collected": self.incorrect_collected,
            "target_fraction": f"{self.target_fraction.numerator}/{self.target_fraction.denominator}"
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (call reset first to ensure state is initialized)
        self.reset()
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # For visualization with Pygame
    pygame.display.set_caption("Fraction Master")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard keys to actions for human play
        action = [0, 0, 0] # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # Only step if an action was taken
        if action[0] != 0 or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we don't need a fixed FPS clock,
        # but a small delay prevents the loop from running too fast on key holds.
        pygame.time.wait(50)
        
    print("Game Over!")
    pygame.quit()