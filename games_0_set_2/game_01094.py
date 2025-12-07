
# Generated: 2025-08-27T16:01:23.574842
# Source Brief: brief_01094.md
# Brief Index: 1094

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear the "
        "highlighted group of blocks. Hold shift to give up."
    )

    game_description = (
        "Clear the grid by selecting groups of same-colored blocks. Larger groups "
        "give more points! Plan your moves carefully before you run out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)

        # Game constants
        self.UI_HEIGHT = 40
        self.GRID_COLS, self.GRID_ROWS = 5, 10
        self.BLOCK_SIZE = min(
            (self.WIDTH) // self.GRID_COLS, (self.HEIGHT - self.UI_HEIGHT) // self.GRID_ROWS
        )
        self.GRID_WIDTH = self.GRID_COLS * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.BLOCK_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = self.UI_HEIGHT
        self.INITIAL_MOVES = 10
        self.NUM_COLORS = 5
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = pygame.Color("#1a1c2c")
        self.COLOR_GRID = pygame.Color("#3f3f74")
        self.COLORS = [
            pygame.Color("#000000"),  # 0: Empty
            pygame.Color("#ff4757"),  # 1: Red
            pygame.Color("#2ed573"),  # 2: Green
            pygame.Color("#1e90ff"),  # 3: Blue
            pygame.Color("#ffa502"),  # 4: Orange
            pygame.Color("#f78fb3"),  # 5: Pink
        ]
        self.COLOR_HIGHLIGHT_FACTOR = 1.5
        self.COLOR_SELECTOR_BORDER = pygame.Color(255, 255, 255)
        self.COLOR_SELECTOR_FILL = pygame.Color(255, 255, 255, 100)
        self.COLOR_TEXT = pygame.Color("#ffffff")
        self.COLOR_FLASH = pygame.Color(255, 255, 255, 200)

        # State variables
        self.grid = None
        self.selector_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.np_random = None
        self.last_cleared_blocks = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = self.np_random.integers(
            1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS)
        )
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_cleared_blocks = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.last_cleared_blocks = []
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            self.game_over = True
            reward = -10.0  # Penalty for giving up
        else:
            self._move_selector(movement)
            if space_held:
                reward += self._clear_blocks()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_board_clear():
                reward += 100.0  # Win bonus
            else:
                reward += -10.0  # Loss penalty for running out of moves

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _move_selector(self, movement):
        if movement == 1:  # Up
            self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 2:  # Down
            self.selector_pos[0] = min(self.GRID_ROWS - 1, self.selector_pos[0] + 1)
        elif movement == 3:  # Left
            self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 4:  # Right
            self.selector_pos[1] = min(self.GRID_COLS - 1, self.selector_pos[1] + 1)

    def _clear_blocks(self):
        row, col = self.selector_pos
        if self.grid[row, col] == 0:
            return 0.0

        connected_blocks = self._find_connected(row, col)
        
        # A move is only consumed if a valid group is selected
        if len(connected_blocks) > 0:
            self.moves_left -= 1
            # sfx: block_select

            # sfx: block_clear_small or block_clear_large based on len
            reward = len(connected_blocks) * 0.1
            if len(connected_blocks) >= 4:
                reward += 1.0

            self.score += len(connected_blocks) ** 2
            
            for r, c in connected_blocks:
                self.grid[r, c] = 0
            self.last_cleared_blocks = connected_blocks

            self._apply_gravity()
            return reward
        
        return 0.0

    def _find_connected(self, r_start, c_start):
        target_color = self.grid[r_start, c_start]
        if target_color == 0:
            return []

        q = [(r_start, c_start)]
        visited = set(q)
        connected = []

        while q:
            r, c = q.pop(0)
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return connected

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            col_blocks = [self.grid[r, c] for r in range(self.GRID_ROWS) if self.grid[r, c] != 0]
            new_col = [0] * (self.GRID_ROWS - len(col_blocks)) + col_blocks
            for r in range(self.GRID_ROWS):
                self.grid[r, c] = new_col[r]

    def _is_board_clear(self):
        return np.all(self.grid == 0)

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        if self._is_board_clear():
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        highlighted_blocks = self._find_connected(self.selector_pos[0], self.selector_pos[1])
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block_rect = pygame.Rect(
                    self.GRID_X + c * self.BLOCK_SIZE,
                    self.GRID_Y + r * self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, block_rect, 1)
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    base_color = self.COLORS[color_idx]
                    if (r, c) in highlighted_blocks:
                        h, s, v, a = base_color.hsva
                        highlight_color = pygame.Color(0, 0, 0)
                        highlight_color.hsva = (h, s, min(100, v * self.COLOR_HIGHLIGHT_FACTOR), a)
                        pygame.draw.rect(self.screen, highlight_color, block_rect)
                    else:
                        pygame.draw.rect(self.screen, base_color, block_rect)

        for r, c in self.last_cleared_blocks:
            flash_rect = pygame.Rect(
                self.GRID_X + c * self.BLOCK_SIZE,
                self.GRID_Y + r * self.BLOCK_SIZE,
                self.BLOCK_SIZE,
                self.BLOCK_SIZE,
            )
            flash_surf = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
            flash_surf.fill(self.COLOR_FLASH)
            self.screen.blit(flash_surf, flash_rect.topleft)

        selector_rect = pygame.Rect(
            self.GRID_X + self.selector_pos[1] * self.BLOCK_SIZE,
            self.GRID_Y + self.selector_pos[0] * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE,
        )
        selector_surf = pygame.Surface(selector_rect.size, pygame.SRCALPHA)
        selector_surf.fill(self.COLOR_SELECTOR_FILL)
        self.screen.blit(selector_surf, selector_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR_BORDER, selector_rect, 3)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, (self.UI_HEIGHT - score_text.get_height()) // 2))

        moves_text = self.font.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(
            moves_text,
            (self.WIDTH - moves_text.get_width() - 10, (self.UI_HEIGHT - moves_text.get_height()) // 2),
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
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
        assert "score" in info
        assert "steps" in info
        assert "moves_left" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Color Grid Cleaner")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    # To prevent rapid-fire actions, we'll only process one action per key press
    last_space_state = 0
    
    while not terminated:
        movement = 0
        space_press = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            # Detect keydown for movement
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
        
        # Check held keys for continuous actions
        keys = pygame.key.get_pressed()
        
        # Check for space bar press (rising edge)
        current_space_state = keys[pygame.K_SPACE]
        if current_space_state and not last_space_state:
            space_press = 1
        last_space_state = current_space_state
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        # Only step if an action is taken
        if movement or space_press or shift_held:
            action = [movement, space_press, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['steps'] > 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}, Terminated: {terminated}")
        else: # If no action, just re-render the current state
            obs = env._get_observation()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over!")
            pygame.time.wait(2000)

        clock.tick(30) 

    env.close()