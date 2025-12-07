
# Generated: 2025-08-28T03:09:14.164484
# Source Brief: brief_01935.md
# Brief Index: 1935

        
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
        "Controls: ↑↓←→ to move. Collect yellow keys and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate procedurally generated mazes, collect keys for points, and reach the exit to win. Maze size increases with success."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.TIME_LIMIT = 60 # In steps, since it's turn-based

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Colors
        self.COLOR_BG = (20, 30, 40) # Dark blue-grey
        self.COLOR_WALL = (44, 62, 80) # Lighter blue-grey
        self.COLOR_PATH = self.COLOR_BG
        self.COLOR_PLAYER = (231, 76, 60) # Bright Red
        self.COLOR_KEY = (241, 196, 15) # Bright Yellow
        self.COLOR_EXIT = (46, 204, 113) # Bright Green
        self.COLOR_TEXT = (236, 240, 241) # White

        # Game state variables
        self.maze_dims = (11, 11) # Must be odd numbers
        self.successful_episodes = 0
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.keys = None
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_status = ""

        # Initialize state for the first time
        # self.reset() is called by the environment runner, but we may need to initialize some vars
        # to pass validation.
        self._initialize_state_for_validation()
        
        # Implementation validation
        self.validate_implementation()

    def _initialize_state_for_validation(self):
        """Initializes a minimal state to pass the validation check in __init__."""
        self.maze_dims = (11, 11)
        self.maze_width, self.maze_height = self.maze_dims
        self.maze = np.zeros((self.maze_height, self.maze_width), dtype=np.uint8)
        self.player_pos = [1, 1]
        self.exit_pos = [self.maze_width - 2, self.maze_height - 2]
        self.keys = []
        self.score = 0
        self.steps = 0
        self.time_remaining = self.TIME_LIMIT
        self.game_over = False

    def _generate_maze(self, width, height):
        """Generates a perfect maze using recursive backtracking."""
        maze = np.ones((height, width), dtype=np.uint8) # 1 = wall
        stack = []
        
        # Start at (1,1)
        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0 # 0 = path
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path to neighbor
                maze[ny, nx] = 0
                maze[(y + ny) // 2, (x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Difficulty scaling
        if self.successful_episodes > 0 and self.successful_episodes % 5 == 0:
            # Increase maze size, but cap it to prevent it from becoming too large for the screen
            new_w = min(self.maze_dims[0] + 2, 51) 
            new_h = min(self.maze_dims[1] + 2, 31)
            self.maze_dims = (new_w, new_h)

        self.maze_width, self.maze_height = self.maze_dims
        self.maze = self._generate_maze(self.maze_width, self.maze_height)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT
        self.game_over = False
        self.win_status = ""

        # Place player, exit, and keys
        self.player_pos = [1, 1]
        self.exit_pos = [self.maze_width - 2, self.maze_height - 2]
        
        # Find all valid floor cells for keys
        floor_cells = np.argwhere(self.maze == 0).tolist()
        
        # Ensure player start and exit are not chosen for keys
        if self.player_pos in floor_cells:
            floor_cells.remove(self.player_pos)
        if self.exit_pos in floor_cells:
            floor_cells.remove(self.exit_pos)

        # Determine number of keys based on progression
        num_keys = 1 + self.successful_episodes // 5
        num_keys = min(num_keys, len(floor_cells), 10) # Cap keys

        key_indices = self.np_random.choice(len(floor_cells), num_keys, replace=False)
        self.keys = [floor_cells[i] for i in key_indices]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1  # Cost per step
        terminated = False

        # --- Update game logic ---
        self.steps += 1
        self.time_remaining -= 1

        # Handle movement
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: # Up
            ny -= 1
        elif movement == 2: # Down
            ny += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Check for valid move
        if self.maze[ny, px] == 0: # It's a path
            self.player_pos = [px, py] = [nx, ny]

        # Check for key collection
        if self.player_pos in self.keys:
            self.keys.remove(self.player_pos)
            self.score += 5
            reward += 5
            # sfx: key collect sound

        # Check for termination conditions
        if self.player_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            if self.time_remaining >= 0:
                self.score += 50
                reward += 50
                self.successful_episodes += 1
                self.win_status = "SUCCESS!"
                # sfx: win sound
            else:
                self.score -= 50
                reward -= 50
                self.win_status = "TOO SLOW!"
                # sfx: fail sound
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_status = "TIME UP!"
            # sfx: fail sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _render_game(self):
        # Calculate cell size and centering offset
        cell_h = self.HEIGHT // self.maze_height
        cell_w = self.WIDTH // self.maze_width
        cell_size = min(cell_w, cell_h)
        
        render_w, render_h = cell_size * self.maze_width, cell_size * self.maze_height
        offset_x = (self.WIDTH - render_w) // 2
        offset_y = (self.HEIGHT - render_h) // 2

        # Draw maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * cell_size,
                        offset_y + y * cell_size,
                        cell_size,
                        cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw keys
        key_radius = int(cell_size * 0.3)
        for kx, ky in self.keys:
            draw_x = int(offset_x + kx * cell_size + cell_size / 2)
            draw_y = int(offset_y + ky * cell_size + cell_size / 2)
            pygame.gfxdraw.filled_circle(self.screen, draw_x, draw_y, key_radius, self.COLOR_KEY)
            pygame.gfxdraw.aacircle(self.screen, draw_x, draw_y, key_radius, self.COLOR_KEY)

        # Draw exit
        exit_radius = int(cell_size * 0.35)
        ex, ey = self.exit_pos
        draw_x = int(offset_x + ex * cell_size + cell_size / 2)
        draw_y = int(offset_y + ey * cell_size + cell_size / 2)
        pygame.gfxdraw.filled_circle(self.screen, draw_x, draw_y, exit_radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, draw_x, draw_y, exit_radius, self.COLOR_EXIT)

        # Draw player
        player_radius = int(cell_size * 0.4)
        px, py = self.player_pos
        draw_x = int(offset_x + px * cell_size + cell_size / 2)
        draw_y = int(offset_y + py * cell_size + cell_size / 2)
        pygame.gfxdraw.filled_circle(self.screen, draw_x, draw_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, draw_x, draw_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_text = self.font_ui.render(f"TIME: {self.time_remaining}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = self.font_game_over.render(self.win_status, True, self.COLOR_TEXT)
            status_rect = status_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(status_text, status_rect)


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
            "time_remaining": self.time_remaining,
            "successful_episodes": self.successful_episodes,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Only register a move on key press, advancing one step
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                    continue # Skip step for this frame
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only step on key presses.
        # The clock tick here just keeps the window responsive.
        env.clock.tick(30)
        
    env.close()