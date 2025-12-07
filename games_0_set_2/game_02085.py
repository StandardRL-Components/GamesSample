import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the robot through the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a procedurally generated maze to reach the green exit. "
        "Win by reaching the exit in under 50 steps. You lose if you hit 10 walls or take 500 steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_COLS, self.MAZE_ROWS = 31, 19  # Odd numbers for maze generation
        self.CELL_SIZE = 20
        self.OFFSET_X = (self.WIDTH - self.MAZE_COLS * self.CELL_SIZE) // 2
        self.OFFSET_Y = (self.HEIGHT - self.MAZE_ROWS * self.CELL_SIZE) // 2

        self.MAX_STEPS = 500
        self.MAX_WALL_HITS = 10
        self.VICTORY_STEP_THRESHOLD = 50

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (45, 55, 65)
        self.COLOR_PATH_TRAVERSED = (30, 38, 45)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_BORDER = (100, 200, 255)
        self.COLOR_EXIT = (0, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_COLLISION = (255, 50, 50)

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables
        self.maze = None
        self.robot_pos = None
        self.start_pos = (1, 1)
        self.exit_pos = (self.MAZE_COLS - 2, self.MAZE_ROWS - 2)
        self.traversed_path = None
        self.steps = 0
        self.score = 0
        self.wall_hits = 0
        self.game_over = False
        self.game_won = False
        self.collision_flash_timer = 0
        
        # self.reset() is called in the __init__ of the parent class,
        # but we need a seed for the maze generation, so we call it again.
        # However, to avoid issues, it's better to let the user call reset.
        # For validation, we'll initialize the state here.
        self.np_random = None # will be seeded in reset
        # self.reset() # This can cause issues, let the user/runner call it.
        
    
    def _generate_maze(self):
        # Create a grid full of walls
        grid = np.ones((self.MAZE_ROWS, self.MAZE_COLS), dtype=np.uint8)
        
        # Use randomized DFS to carve paths
        stack = []
        
        # Start carving from the start position
        x, y = self.start_pos
        grid[y, x] = 0
        stack.append((x, y))
        
        while stack:
            cx, cy = stack[-1]
            
            # Find neighbors (2 cells away)
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_COLS-1 and 0 < ny < self.MAZE_ROWS-1 and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Choose a random neighbor
                chosen_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[chosen_index]
                
                # Carve wall between current and neighbor
                grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                grid[ny, nx] = 0
                
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure start and exit are open
        grid[self.start_pos[1], self.start_pos[0]] = 0
        grid[self.exit_pos[1], self.exit_pos[0]] = 0
        
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.maze = self._generate_maze()
        self.robot_pos = self.start_pos
        self.traversed_path = {self.robot_pos}
        
        self.steps = 0
        self.score = 0
        self.wall_hits = 0
        self.game_over = False
        self.game_won = False
        self.collision_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Cost for taking a step

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update robot position based on movement
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            next_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            
            # Check for wall collision
            if self.maze[next_pos[1], next_pos[0]] == 1:
                # Wall hit
                self.wall_hits += 1
                reward -= 1.0
                self.collision_flash_timer = 3  # Flash for 3 frames
                # sfx: wall_thud.wav
            else:
                # Successful move
                self.robot_pos = next_pos
                self.traversed_path.add(self.robot_pos)

        self.steps += 1
        
        # Check for termination conditions
        terminated = False
        if self.robot_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            self.game_won = True
            if self.steps < self.VICTORY_STEP_THRESHOLD:
                reward += 100.0  # Victory bonus
                # sfx: victory_fast.wav
            else:
                reward += 10.0  # Reached exit bonus
                # sfx: victory_slow.wav
        
        if self.wall_hits >= self.MAX_WALL_HITS:
            terminated = True
            self.game_over = True
            # sfx: game_over_lose.wav

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # In this game, truncation is also a form of termination
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw traversed path
        for pos in self.traversed_path:
            rect = pygame.Rect(
                self.OFFSET_X + pos[0] * self.CELL_SIZE,
                self.OFFSET_Y + pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_PATH_TRAVERSED, rect)

        # Draw maze walls
        for r in range(self.MAZE_ROWS):
            for c in range(self.MAZE_COLS):
                if self.maze[r, c] == 1:
                    rect = pygame.Rect(
                        self.OFFSET_X + c * self.CELL_SIZE,
                        self.OFFSET_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw exit
        exit_rect = pygame.Rect(
            self.OFFSET_X + self.exit_pos[0] * self.CELL_SIZE,
            self.OFFSET_Y + self.exit_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw robot
        robot_center_x = self.OFFSET_X + self.robot_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        robot_center_y = self.OFFSET_Y + self.robot_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        robot_size = self.CELL_SIZE * 0.7
        robot_rect = pygame.Rect(
            robot_center_x - robot_size / 2,
            robot_center_y - robot_size / 2,
            robot_size, robot_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_BORDER, robot_rect, width=2, border_radius=3)

    def _render_ui(self):
        # Render steps
        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))

        # Render wall hits
        hits_text = self.font_ui.render(f"Wall Hits: {self.wall_hits}/{self.MAX_WALL_HITS}", True, self.COLOR_TEXT)
        hits_rect = hits_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(hits_text, hits_rect)

        # Render collision flash
        if self.collision_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.collision_flash_timer / 3.0))
            flash_surface.fill((*self.COLOR_COLLISION, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.collision_flash_timer -= 1

        # Render game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                msg = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_COLLISION
            
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        if self.maze is not None:
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
            "wall_hits": self.wall_hits,
            "robot_pos": self.robot_pos,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Starting implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8, f"Obs dtype is {obs.dtype}"
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # To do so, you must comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Example of creating and running the environment
    env = GameEnv()
    env.validate_implementation() # Run validation
    obs, info = env.reset(seed=123)
    
    # To render, you would need a display. The following is for interactive testing.
    # For agent training, you would just call env.step() and env.reset()
    try:
        # Create a display window
        pygame.display.set_caption("Maze Robot")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        running = True
    except pygame.error:
        print("Pygame display could not be initialized (likely in headless mode). Running a short test loop.")
        running = False
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: Reward={reward}, Terminated={terminated}, Truncated={truncated}")
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset(seed=123+i+1)
        
    terminated = False
    truncated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 # Press 'r' to reset
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = np.array([movement, space_held, shift_held])
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The game only advances on action. We add a small delay to prevent
        # the loop from running too fast and consuming 100% CPU.
        pygame.time.wait(30)

    env.close()