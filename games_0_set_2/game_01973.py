
# Generated: 2025-08-28T03:16:01.480827
# Source Brief: brief_01973.md
# Brief Index: 1973

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (up, down, left, right) to move the robot. "
        "Reach the green exit before you run out of steps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated maze to the escape zone "
        "within a limited number of steps. Each move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (255, 255, 255)  # White
    COLOR_GRID = (220, 220, 220)  # Light Grey
    COLOR_OBSTACLE = (20, 20, 20)  # Near Black
    COLOR_ROBOT = (255, 50, 50)  # Bright Red
    COLOR_ROBOT_GLOW = (255, 100, 100) # Lighter Red
    COLOR_EXIT = (50, 255, 50)  # Bright Green
    COLOR_PATH = (50, 100, 255)  # Bright Blue
    COLOR_UI_TEXT = (10, 10, 10)  # Dark Grey

    # Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE  # 16
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE  # 10

    # Game rules
    INITIAL_OBSTACLES = 15
    STEP_LIMIT = 30
    MAX_EPISODE_STEPS = 1000 # Gym standard, but game ends sooner

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game state variables that persist across resets
        self.obstacle_count = self.INITIAL_OBSTACLES
        
        # Initialize other state variables
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.obstacles = set()
        self.path_taken = []
        self.steps_remaining = 0
        self.total_steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()

        self.steps_remaining = self.STEP_LIMIT
        self.total_steps = 0
        self.score = 0
        self.game_over = False
        self.path_taken = [self.robot_pos]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        """
        Generates a new maze layout ensuring a path from start to exit.
        1. Place robot and exit.
        2. Place obstacles randomly.
        3. Use BFS to check for a path.
        4. If no path, "punch holes" in obstacles bordering the reachable area until a path is found.
        """
        all_cells = set(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )

        # 1. Place robot and exit
        start_end_indices = self.np_random.choice(len(all_cells), size=2, replace=False)
        all_cells_list = list(all_cells)
        self.robot_pos = all_cells_list[start_end_indices[0]]
        self.exit_pos = all_cells_list[start_end_indices[1]]

        possible_obstacle_cells = list(all_cells - {self.robot_pos, self.exit_pos})
        
        # 2. Place obstacles
        num_obstacles = min(self.obstacle_count, len(possible_obstacle_cells))
        obstacle_indices = self.np_random.choice(
            len(possible_obstacle_cells), size=num_obstacles, replace=False
        )
        self.obstacles = {possible_obstacle_cells[i] for i in obstacle_indices}

        # 3. Ensure path exists
        while not self._is_path_available():
            # 4. Punch a hole
            q = deque([self.robot_pos])
            visited = {self.robot_pos}
            border_obstacles = set()

            while q:
                x, y = q.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (x + dx, y + dy)
                    if self._is_in_bounds(neighbor):
                        if neighbor in self.obstacles:
                            border_obstacles.add(neighbor)
                        elif neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
            
            if not border_obstacles: # Should not happen, but a safeguard
                self.obstacles.clear()
                break
                
            obstacle_to_remove = random.choice(list(border_obstacles))
            self.obstacles.remove(obstacle_to_remove)

    def _is_in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT

    def _is_path_available(self):
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        while q:
            pos = q.popleft()
            if pos == self.exit_pos:
                return True
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if self._is_in_bounds(neighbor) and neighbor not in self.obstacles and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        if movement != 0:
            target_pos = list(self.robot_pos)
            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right
            
            target_pos = tuple(target_pos)

            if self._is_in_bounds(target_pos) and target_pos not in self.obstacles:
                self.robot_pos = target_pos
                self.path_taken.append(self.robot_pos)

        self.total_steps += 1
        self.steps_remaining -= 1

        reward = -0.1  # Cost for taking a step
        terminated = False

        if self.robot_pos == self.exit_pos:
            reward += 10.0
            self.score += 1
            terminated = True
            self.game_over = True
            # Increase difficulty for the next game
            self.obstacle_count = min(self.obstacle_count + 2, (self.GRID_WIDTH * self.GRID_HEIGHT) - 5)
        elif self.steps_remaining <= 0:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _grid_to_pixel(self, pos):
        x, y = pos
        return (x * self.CELL_SIZE, y * self.CELL_SIZE)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw path taken
        if len(self.path_taken) > 1:
            for i in range(len(self.path_taken) - 1):
                start_px = self._grid_to_pixel(self.path_taken[i])
                end_px = self._grid_to_pixel(self.path_taken[i+1])
                center_start = (start_px[0] + self.CELL_SIZE // 2, start_px[1] + self.CELL_SIZE // 2)
                center_end = (end_px[0] + self.CELL_SIZE // 2, end_px[1] + self.CELL_SIZE // 2)
                
                alpha = int(100 + 155 * (i / max(1, len(self.path_taken) - 1)))
                color = (*self.COLOR_PATH, alpha)
                pygame.gfxdraw.line(self.screen, center_start[0], center_start[1], center_end[0], center_end[1], color)

        # Draw obstacles
        for obs_pos in self.obstacles:
            px, py = self._grid_to_pixel(obs_pos)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_px, exit_py, self.CELL_SIZE, self.CELL_SIZE))

        # Draw robot with glow
        robot_px, robot_py = self._grid_to_pixel(self.robot_pos)
        center_x, center_y = robot_px + self.CELL_SIZE // 2, robot_py + self.CELL_SIZE // 2
        
        glow_radius = int(self.CELL_SIZE * 0.6)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_ROBOT_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (center_x - glow_radius, center_y - glow_radius))

        robot_size = int(self.CELL_SIZE * 0.8)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (
            center_x - robot_size // 2, center_y - robot_size // 2, robot_size, robot_size
        ))

    def _render_ui(self):
        text_surface = self.font.render(f"Steps: {self.steps_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 5))
        
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "steps_remaining": self.steps_remaining,
        }

    def close(self):
        pygame.font.quit()
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
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    print("\n--- Game Reset ---")
                    continue
                
                if not done:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                        if terminated:
                            print("Game Over! Press 'R' to reset.")
                            done = True

        display_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(display_surface, (0, 0))
        pygame.display.flip()
        
    env.close()