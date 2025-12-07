
# Generated: 2025-08-28T06:15:42.277240
# Source Brief: brief_02883.md
# Brief Index: 2883

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based maze navigation game where a robot must find the goal.

    The game is turn-based. Each step, the agent chooses a direction to move.
    The goal is to reach the green square (goal) while avoiding blue squares (obstacles).
    A reward of +10 is given for reaching the goal, and a penalty of -10 for
    colliding with an obstacle. Each step incurs a small penalty of -0.1.

    The maze is procedurally generated at the start of each episode, and it is
    guaranteed to be solvable. The number of obstacles increases every 5 successful
    episodes to scale the difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string:
    user_guide = (
        "Controls: Use ↑↓←→ to move the robot. Reach the green goal while avoiding blue obstacles."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a procedurally generated grid-based maze to reach the goal in the fewest steps possible."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 1000

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 22, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 28)

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (40, 40, 40)
        self.COLOR_ROBOT = (255, 70, 70)
        self.COLOR_OBSTACLE = (70, 130, 255)
        self.COLOR_GOAL = (70, 255, 70)
        self.COLOR_TEXT = (220, 220, 220)

        # Persistent State (across resets)
        self.base_num_obstacles = 10
        self.successful_episodes_count = 0
        self.score = 0  # Total successful completions

        # Episode-specific state variables (initialized in reset)
        self.steps = 0
        self.game_over = False
        self.robot_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.obstacle_positions = set()
        self.num_obstacles = 0
        
        # This will be called once to initialize the first game state
        # self.reset() is called below, so no need to call it here.

        # Run validation check
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset episode state
        self.steps = 0
        self.game_over = False

        # Update difficulty based on persistent success count
        self.num_obstacles = self.base_num_obstacles + (self.successful_episodes_count // 5)
        
        # Generate a new, solvable maze
        self._generate_maze()

        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        """Generates a new maze layout, ensuring a path from start to goal exists."""
        while True:
            all_cells = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))
            
            # 1. Pick start and goal, ensuring they are not the same
            self.robot_pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            self.goal_pos = self.robot_pos
            while self.robot_pos == self.goal_pos:
                self.goal_pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))

            # 2. Pick obstacle locations
            possible_obstacle_cells = list(all_cells - {self.robot_pos, self.goal_pos})
            
            num_to_pick = min(self.num_obstacles, len(possible_obstacle_cells))
            indices = self.np_random.choice(len(possible_obstacle_cells), num_to_pick, replace=False)
            self.obstacle_positions = {possible_obstacle_cells[i] for i in indices}
            
            # 3. Verify path exists using Breadth-First Search (BFS)
            if self._path_exists():
                break # Maze is valid and solvable

    def _path_exists(self):
        """Uses BFS to check for a path from robot_pos to goal_pos."""
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        
        while q:
            current_pos = q.popleft()

            if current_pos == self.goal_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                    continue
                
                if next_pos in visited or next_pos in self.obstacle_positions:
                    continue
                
                visited.add(next_pos)
                q.append(next_pos)
        
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        movement = action[0]
        # space_held (action[1]) and shift_held (action[2]) are ignored.
        
        self.steps += 1
        reward = -0.1  # Small penalty for each step taken

        # --- Update game logic ---
        dx, dy = 0, 0
        if movement == 1: dy = -1   # Up
        elif movement == 2: dy = 1    # Down
        elif movement == 3: dx = -1   # Left
        elif movement == 4: dx = 1    # Right
        
        if movement != 0: # Only move if a movement action is given
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            
            # Check boundaries and update position if valid
            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                self.robot_pos = new_pos
                # Sound placeholder: # play_sound('move')

        # --- Check for termination conditions ---
        terminated = False
        if self.robot_pos in self.obstacle_positions:
            reward = -10.0
            terminated = True
            # Sound placeholder: # play_sound('collision')
        elif self.robot_pos == self.goal_pos:
            reward = 10.0
            terminated = True
            self.successful_episodes_count += 1
            self.score += 1
            # Sound placeholder: # play_sound('win')
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_W, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

        # Draw obstacles
        for ox, oy in self.obstacle_positions:
            rect = pygame.Rect(ox * self.CELL_SIZE, oy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)

        # Draw goal
        gx, gy = self.goal_pos
        goal_rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect)

        # Draw robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(rx * self.CELL_SIZE, ry * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)
        # Add a highlight for better visibility
        inner_rect = robot_rect.inflate(-self.CELL_SIZE * 0.5, -self.CELL_SIZE * 0.5)
        highlight_color = tuple(min(255, c + 80) for c in self.COLOR_ROBOT)
        pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=2)

    def _render_ui(self):
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        score_text = self.font.render(f"Solved: {self.score}", True, self.COLOR_TEXT)
        
        self.screen.blit(steps_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_W - score_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")