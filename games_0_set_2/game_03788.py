
# Generated: 2025-08-28T00:25:02.054625
# Source Brief: brief_03788.md
# Brief Index: 3788

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Find the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to find the exit within a limited number of steps. "
        "Each step costs points, but reaching the exit gives a large bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    MAZE_W, MAZE_H = 15, 15
    GRID_W, GRID_H = MAZE_W * 2 + 1, MAZE_H * 2 + 1
    MAX_STEPS = 75
    SCREEN_W, SCREEN_H = 640, 400

    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (80, 80, 90)
    COLOR_PATH = (40, 40, 55)
    COLOR_TRAIL = (60, 90, 120)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_EXIT = (0, 255, 120)
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Calculate rendering geometry
        render_area_size = self.SCREEN_H - 40 # Leave padding for UI
        self.cell_size = render_area_size // self.GRID_H
        self.maze_render_size_w = self.cell_size * self.GRID_W
        self.maze_render_size_h = self.cell_size * self.GRID_H
        self.offset_x = (self.SCREEN_W - self.maze_render_size_w) // 2
        self.offset_y = (self.SCREEN_H - self.maze_render_size_h) // 2

        # Game state variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_trail = []
        
        self.validate_implementation()
    
    def _generate_maze(self):
        # 1 = wall, 0 = path
        maze = np.ones((self.GRID_H, self.GRID_W), dtype=np.uint8)
        
        # Use recursive backtracking (iterative version with a stack)
        stack = []
        visited = set()

        # Start cell in maze coordinates (0-14)
        start_x, start_y = self.np_random.integers(0, self.MAZE_W), self.np_random.integers(0, self.MAZE_H)
        
        stack.append((start_x, start_y))
        visited.add((start_x, start_y))
        maze[start_y * 2 + 1, start_x * 2 + 1] = 0

        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                idx = self.np_random.choice(len(neighbors))
                nx, ny = neighbors[idx]

                # Carve path to neighbor (wall is between current and next cell)
                maze[(cy * 2 + 1) + (ny - cy), (cx * 2 + 1) + (nx - cx)] = 0
                maze[ny * 2 + 1, nx * 2 + 1] = 0

                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop() # Backtrack

        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        
        # Find all valid path cells
        path_cells = np.argwhere(self.maze == 0)
        
        # Player start
        start_idx = self.np_random.choice(len(path_cells))
        self.player_pos = tuple(path_cells[start_idx][::-1]) # Use (x, y) format
        
        # Find an exit point far from the start
        distances = np.linalg.norm(path_cells - path_cells[start_idx], axis=1)
        # Add some randomness to the farthest point selection
        farthest_indices = np.argsort(distances)[-10:]
        exit_idx = self.np_random.choice(farthest_indices)
        self.exit_pos = tuple(path_cells[exit_idx][::-1])

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_trail = [self.player_pos]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        px, py = self.player_pos
        nx, ny = px, py
        wall_x, wall_y = px, py

        # Calculate target position and the wall in between
        if movement == 1:  # Up
            ny, wall_y = py - 2, py - 1
        elif movement == 2:  # Down
            ny, wall_y = py + 2, py + 1
        elif movement == 3:  # Left
            nx, wall_x = px - 2, px - 1
        elif movement == 4:  # Right
            nx, wall_x = px + 2, px + 1

        moved = False
        if movement != 0:
            # Check boundaries and wall collision
            if 0 <= wall_x < self.GRID_W and 0 <= wall_y < self.GRID_H and self.maze[wall_y, wall_x] == 0:
                self.player_pos = (nx, ny)
                self.path_trail.append(self.player_pos)
                moved = True
        
        self.steps += 1
        
        # --- Calculate Reward ---
        reward = -0.1  # Cost for taking a step

        if moved:
            # Check for risky/safe moves based on the number of exits from the new cell
            pos_x, pos_y = self.player_pos
            open_neighbors = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_x, check_y = pos_x + dx, pos_y + dy
                if 0 <= check_x < self.GRID_W and 0 <= check_y < self.GRID_H and self.maze[check_y, check_x] == 0:
                    open_neighbors += 1
            
            if open_neighbors == 1: # Moved into a dead end
                reward += 5.0
            elif open_neighbors > 2: # Moved into an intersection
                reward -= 0.2

        # --- Check Termination ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw maze background (paths and walls)
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(
                    self.offset_x + x * self.cell_size,
                    self.offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw path trail
        for pos in self.path_trail:
            rect = pygame.Rect(
                self.offset_x + pos[0] * self.cell_size,
                self.offset_y + pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.COLOR_TRAIL, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            self.offset_x + ex * self.cell_size,
            self.offset_y + ey * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.offset_x + px * self.cell_size,
            self.offset_y + py * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Add a bright core to the player for visibility
        pygame.draw.rect(self.screen, (200, 220, 255), player_rect.inflate(-self.cell_size//2, -self.cell_size//2))

    def _render_ui(self):
        steps_left = self.MAX_STEPS - self.steps
        score_text = f"SCORE: {self.score:.1f}"
        steps_text = f"STEPS LEFT: {max(0, steps_left)}"
        
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        steps_surf = self.font.render(steps_text, True, self.COLOR_TEXT)
        
        self.screen.blit(steps_surf, (20, 10))
        self.screen.blit(score_surf, (self.SCREEN_W - score_surf.get_width() - 20, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset first to initialize everything
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=1)
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

if __name__ == '__main__':
    import time
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window for interactive play
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    print("Press 'R' to reset the maze.")
    
    # Game loop for human play
    while running:
        action = 0 # No-op by default
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Environment Reset ---")
                    continue
                
                # Since auto_advance is False, we only step on key press
                # The action is a MultiDiscrete, so we build the full action array
                full_action = [action, 0, 0] # space and shift are not used
                
                obs, reward, terminated, truncated, info = env.step(full_action)
                total_reward += reward
                
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

                if terminated:
                    print("--- Episode Finished ---")
                    time.sleep(1) # Pause before auto-reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- New Episode Started ---")

        # Update the display with the latest observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()