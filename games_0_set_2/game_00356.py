
# Generated: 2025-08-27T13:23:53.419283
# Source Brief: brief_00356.md
# Brief Index: 356

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the robot. Plan your path to the green goal without running out of energy."
    )

    game_description = (
        "Program a robot's path through a procedurally generated maze to reach the goal while managing its limited energy."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        self.MAX_LEVEL = 10 # Cap difficulty scaling

        # Visuals
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (80, 90, 100)
        self.COLOR_ROBOT = (0, 200, 255)
        self.COLOR_ROBOT_OUTLINE = (200, 255, 255)
        self.COLOR_GOAL = (0, 255, 150)
        self.COLOR_PATH = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_ENERGY_BG = (40, 50, 60)
        self.COLOR_ENERGY_FILL = (0, 150, 255)
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables that persist across resets
        self.successful_episodes = 0
        
        # Initialize state variables
        self.grid_w = 0
        self.grid_h = 0
        self.cell_size = 0
        self.offset_x = 0
        self.offset_y = 0
        self.maze_grid = []
        self.robot_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.max_energy = 1
        self.energy = 1
        self.path_history = deque(maxlen=50)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def _generate_maze(self):
        level = min(self.successful_episodes // 5, self.MAX_LEVEL)
        
        base_w, base_h = 20, 15
        self.grid_w = base_w + int(base_w * 0.1 * level)
        self.grid_h = base_h + int(base_h * 0.1 * level)
        num_obstacles = 10 + level * 2

        self.cell_size = min(
            (self.SCREEN_WIDTH - 40) // self.grid_w,
            (self.SCREEN_HEIGHT - 80) // self.grid_h
        )
        self.offset_x = (self.SCREEN_WIDTH - self.grid_w * self.cell_size) // 2
        self.offset_y = (self.SCREEN_HEIGHT - self.grid_h * self.cell_size) // 2 + 20

        while True:
            grid = np.zeros((self.grid_w, self.grid_h), dtype=np.uint8)
            
            start_pos = (1, self.np_random.integers(1, self.grid_h - 1))
            end_pos = (self.grid_w - 2, self.np_random.integers(1, self.grid_h - 1))
            
            # Place obstacles
            for _ in range(num_obstacles):
                x, y = self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h)
                if (x, y) != start_pos and (x, y) != end_pos:
                    grid[x, y] = 1

            # Check for solvability using BFS
            queue = deque([start_pos])
            visited = {start_pos}
            path_found = False
            while queue:
                x, y = queue.popleft()
                if (x, y) == end_pos:
                    path_found = True
                    break
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and grid[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            
            if path_found:
                self.maze_grid = grid
                self.robot_pos = start_pos
                self.goal_pos = end_pos
                self.max_energy = int((self.grid_w + self.grid_h) * 1.5)
                return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_history.clear()

        self._generate_maze()
        
        self.energy = self.max_energy
        self.path_history.append(self.robot_pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -1.0
        
        moved = False
        if movement != 0: # 0 is no-op
            rx, ry = self.robot_pos
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            nx, ny = rx + dx, ry + dy
            
            # Consume energy for attempting a move
            self.energy -= 1

            # Check boundaries and walls
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and self.maze_grid[nx, ny] == 0:
                self.robot_pos = (nx, ny)
                moved = True

        if moved:
            self.path_history.append(self.robot_pos)

        self.steps += 1
        terminated = False
        
        # Check termination conditions
        if self.robot_pos == self.goal_pos:
            reward = 100.0
            self.score += 100
            terminated = True
            self.successful_episodes += 1
        elif self.energy <= 0:
            reward = -100.0
            self.score -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render path trail
        for i, pos in enumerate(self.path_history):
            alpha = int(255 * (i / len(self.path_history)))
            color = (*self.COLOR_PATH, alpha)
            center_x = self.offset_x + int((pos[0] + 0.5) * self.cell_size)
            center_y = self.offset_y + int((pos[1] + 0.5) * self.cell_size)
            radius = self.cell_size // 4
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (center_x - radius, center_y - radius))

        # Render walls and goal
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                rect = pygame.Rect(
                    self.offset_x + x * self.cell_size,
                    self.offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                if self.maze_grid[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif (x, y) == self.goal_pos:
                    pygame.draw.rect(self.screen, self.COLOR_GOAL, rect)
        
        # Render robot
        robot_rect = pygame.Rect(
            self.offset_x + self.robot_pos[0] * self.cell_size,
            self.offset_y + self.robot_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect.inflate(-2, -2))
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_OUTLINE, robot_rect, 1)

    def _render_ui(self):
        # Energy Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 20
        
        energy_ratio = max(0, self.energy / self.max_energy)
        fill_width = int(bar_width * energy_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_ENERGY_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        
        energy_text = self.font_main.render(f"ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (bar_x - energy_text.get_width() - 10, bar_y))
        
        # Score Text
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        # Level Text
        level = self.successful_episodes // 5
        level_text = self.font_main.render(f"LEVEL: {level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 20))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.robot_pos == self.goal_pos:
                msg = "GOAL REACHED!"
                color = self.COLOR_GOAL
            else:
                msg = "OUT OF ENERGY"
                color = (255, 80, 80)
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "level": self.successful_episodes // 5
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        
        # Test game-specific assertions
        assert 0 <= self.robot_pos[0] < self.grid_w
        assert 0 <= self.robot_pos[1] < self.grid_h
        assert self.energy >= 0

        print("✓ Implementation validated successfully")