
# Generated: 2025-08-28T04:49:47.514976
# Source Brief: brief_02429.md
# Brief Index: 2429

        
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
        "Controls: Arrow keys to move the robot. Reach the green exit before running out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Navigate a robot through a procedurally generated maze to reach the exit. Each move costs, so find the most efficient path!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        
        # Visual constants
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (5, 5, 10)
        self.COLOR_PATH = (210, 210, 220)
        self.COLOR_ROBOT = (255, 60, 60)
        self.COLOR_ROBOT_GLOW = (255, 120, 120)
        self.COLOR_EXIT = (60, 255, 60)
        self.COLOR_EXIT_GLOW = (120, 255, 120)
        self.COLOR_GRID_LINE = (40, 40, 60)
        self.COLOR_TEXT = (240, 240, 240)

        # Calculate grid rendering properties
        self.CELL_SIZE = 36
        self.GRID_AREA_SIZE = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_TOP_LEFT_X = (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_TOP_LEFT_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) // 2

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.grid = None
        self.robot_pos = None
        self.exit_pos = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _generate_maze(self):
        """Generates a maze using randomized DFS and ensures a valid path exists."""
        while True:
            # 0 for path, 1 for wall
            grid = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
            stack = deque()
            
            # Start DFS from a random cell
            start_x, start_y = self.np_random.integers(0, self.GRID_SIZE, size=2)
            grid[start_y, start_x] = 0
            stack.append((start_x, start_y))

            while stack:
                x, y = stack[-1]
                neighbors = []
                for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and grid[ny, nx] == 1:
                        neighbors.append((nx, ny))
                
                if neighbors:
                    nx, ny = random.choice(neighbors)
                    grid[ny, nx] = 0
                    grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                    stack.append((nx, ny))
                else:
                    stack.pop()

            # Select start and end points
            possible_starts = np.argwhere(grid == 0)
            if len(possible_starts) < 2: continue

            start_idx, end_idx = self.np_random.choice(len(possible_starts), 2, replace=False)
            robot_pos = tuple(possible_starts[start_idx])
            exit_pos = tuple(possible_starts[end_idx])

            # Check path length
            path_length = self._find_shortest_path(grid, robot_pos, exit_pos)
            if 1 < path_length <= self.MAX_MOVES:
                return grid, (robot_pos[1], robot_pos[0]), (exit_pos[1], exit_pos[0])

    def _find_shortest_path(self, grid, start, end):
        """BFS to find shortest path length."""
        q = deque([(start, 0)])
        visited = {tuple(start)}
        while q:
            (r, c), dist = q.popleft()
            if (r, c) == tuple(end):
                return dist
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))
        return -1 # No path found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES

        # Generate a valid maze
        self.grid, self.robot_pos, self.exit_pos = self._generate_maze()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1 # Cost of taking a step
        
        # Update game logic
        moved = False
        if movement != 0: # 0 is no-op
            self.moves_left -= 1
            
            x, y = self.robot_pos
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            
            nx, ny = x + dx, y + dy
            
            # Check boundaries and walls
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny, nx] == 0:
                self.robot_pos = (nx, ny)
                moved = True
        
        self.steps += 1
        
        # Check termination conditions
        if self.robot_pos == self.exit_pos:
            reward += 100.0 # Reached exit
            self.game_over = True
        elif self.moves_left <= 0:
            self.game_over = True
        
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Draw grid cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.GRID_TOP_LEFT_X + c * self.CELL_SIZE,
                    self.GRID_TOP_LEFT_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                color = self.COLOR_PATH if self.grid[r, c] == 0 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, cell_rect)
        
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                             (self.GRID_TOP_LEFT_X + i * self.CELL_SIZE, self.GRID_TOP_LEFT_Y),
                             (self.GRID_TOP_LEFT_X + i * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + self.GRID_AREA_SIZE))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                             (self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y + i * self.CELL_SIZE),
                             (self.GRID_TOP_LEFT_X + self.GRID_AREA_SIZE, self.GRID_TOP_LEFT_Y + i * self.CELL_SIZE))

        # Draw exit
        exit_x, exit_y = self.exit_pos
        exit_center_x = self.GRID_TOP_LEFT_X + int((exit_x + 0.5) * self.CELL_SIZE)
        exit_center_y = self.GRID_TOP_LEFT_Y + int((exit_y + 0.5) * self.CELL_SIZE)
        exit_size = int(self.CELL_SIZE * 0.7)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, 
                         (exit_center_x - exit_size//2, exit_center_y - exit_size//2, exit_size, exit_size), 
                         border_radius=4)
        
        # Draw robot
        robot_x, robot_y = self.robot_pos
        robot_center_x = self.GRID_TOP_LEFT_X + int((robot_x + 0.5) * self.CELL_SIZE)
        robot_center_y = self.GRID_TOP_LEFT_Y + int((robot_y + 0.5) * self.CELL_SIZE)
        robot_size = int(self.CELL_SIZE * 0.8)
        
        # Glow effect for robot
        glow_radius = robot_size // 2 + 5
        for i in range(glow_radius, 0, -2):
            alpha = 100 * (1 - i / glow_radius)
            s = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_ROBOT_GLOW, alpha), (i, i), i)
            self.screen.blit(s, (robot_center_x - i, robot_center_y - i), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_ROBOT, 
                         (robot_center_x - robot_size//2, robot_center_y - robot_size//2, robot_size, robot_size), 
                         border_radius=4)

    def _render_ui(self):
        # Render Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Render Score
        score_text = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.robot_pos == self.exit_pos:
                msg = "SUCCESS!"
                color = self.COLOR_EXIT
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_ROBOT

            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


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
            "moves_left": self.moves_left,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Example ---
    # This demonstrates how a human would play the game.
    # Pygame window setup for visualization.
    
    # Re-initialize pygame to create a display window
    pygame.quit()
    pygame.init()
    pygame.display.set_caption("Maze Robot")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Manual Play ---")
    print(env.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
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
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q: # Quit on 'q'
                    running = False
                
                if action[0] != 0: # If a move key was pressed
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.1f}, Done: {done}, Info: {info}")
                    if done:
                        print("Episode finished. Press 'r' to reset or 'q' to quit.")

        # Draw the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if not running:
            break

    env.close()
    print("Manual play finished.")