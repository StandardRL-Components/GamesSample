
# Generated: 2025-08-28T03:31:45.500580
# Source Brief: brief_02055.md
# Brief Index: 2055

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based maze game where a robot navigates to an exit while avoiding pits.
    The maze is procedurally generated for each episode, with a guaranteed path to the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one square at a time."

    # Short, user-facing description of the game
    game_description = "Navigate a robot through a procedurally generated maze to reach the green exit, avoiding black pits. You get a large bonus for reaching the exit within 50 steps."

    # Frames advance only when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 500 # Hard step limit for episode termination
        self.REWARD_STEP_LIMIT = 50 # Soft limit for best reward

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (220, 220, 220)
        self.COLOR_PIT = (0, 0, 0)
        self.COLOR_ROBOT = (255, 60, 60)
        self.COLOR_ROBOT_BORDER = (180, 40, 40)
        self.COLOR_EXIT = (60, 255, 60)
        self.COLOR_EXIT_BORDER = (40, 180, 40)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font = pygame.font.SysFont("sans", 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.pit_locations = set()

        # Initialize state variables for the first time
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Uncomment to run validation on init

    def _generate_maze(self):
        """
        Generates a new maze layout, ensuring a path exists from start to exit.
        """
        self.start_pos = (1, self.GRID_ROWS - 2)
        self.exit_pos = (self.GRID_COLS - 2, 1)

        while True:
            self.pit_locations = set()
            num_pits = self.np_random.integers(low=25, high=40)
            
            for _ in range(num_pits):
                px, py = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
                if (px, py) != self.start_pos and (px, py) != self.exit_pos:
                    self.pit_locations.add((px, py))

            if self._has_path(self.start_pos, self.exit_pos):
                break # Found a valid maze

        self.robot_pos = self.start_pos

    def _has_path(self, start, end):
        """
        Checks for a path using Breadth-First Search (BFS).
        """
        q = deque([start])
        visited = {start}
        
        while q:
            cx, cy = q.popleft()

            if (cx, cy) == end:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and (nx, ny) not in self.pit_locations:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        # space_held and shift_held are ignored as per the brief
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = -0.1  # Small penalty for each step taken

        # --- Handle Movement ---
        prev_pos = self.robot_pos
        px, py = self.robot_pos
        
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        # movement == 0 is no-op

        # Check boundaries
        if 0 <= px < self.GRID_COLS and 0 <= py < self.GRID_ROWS:
            self.robot_pos = (px, py)

        # --- Check for Terminal Conditions ---
        terminated = False
        if self.robot_pos == self.exit_pos:
            if self.steps <= self.REWARD_STEP_LIMIT:
                reward += 100.0
            else:
                reward += 50.0
            terminated = True
        elif self.robot_pos in self.pit_locations:
            reward += -10.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

        # Draw pits
        for px, py in self.pit_locations:
            rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PIT, rect)

        # Draw exit
        ex, ey = self.exit_pos
        border_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = border_rect.inflate(-self.CELL_SIZE * 0.2, -self.CELL_SIZE * 0.2)
        pygame.draw.rect(self.screen, self.COLOR_EXIT_BORDER, border_rect)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, inner_rect)
        
        # Draw robot
        rx, ry = self.robot_pos
        border_rect = pygame.Rect(rx * self.CELL_SIZE, ry * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = border_rect.inflate(-self.CELL_SIZE * 0.2, -self.CELL_SIZE * 0.2)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_BORDER, border_rect)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, inner_rect)

    def _render_ui(self):
        # Render step count
        step_text = self.font.render(f"Steps: {self.steps}/{self.REWARD_STEP_LIMIT}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (10, 10))

        # Render score
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    
    # --- Pygame setup for interactive play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Maze Robot")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                
                # Since auto_advance is False, we step on key press
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                    if terminated:
                        print("--- Episode Finished --- Press 'R' to reset.")
                        
        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for interactive mode
        
    env.close()