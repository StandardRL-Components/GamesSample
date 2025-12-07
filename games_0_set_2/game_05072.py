import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a procedurally generated maze game.

    The player must navigate a red square from the top-left to the green exit
    at the bottom-right before the timer runs out. The maze size increases
    with each successful completion.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Reach the green exit before the timer runs out."
    )

    game_description = (
        "Navigate a procedurally generated maze to reach the exit within a time limit. "
        "The maze gets larger each time you succeed."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the maze game environment.
        """
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game Configuration ---
        self.max_steps = 1000
        self.initial_maze_dim = 7  # 7x7 cells -> 15x15 grid
        self.max_maze_dim = 49  # 49x49 cells -> 99x99 grid
        self.maze_level_increment = 2  # Add 2 cells (4 grid units) per level

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)       # Dark blue/black for path
        self.COLOR_WALL = (220, 220, 240)  # Bright white/light grey for walls
        self.COLOR_PLAYER = (255, 50, 50)  # Bright red
        self.COLOR_EXIT = (50, 255, 50)    # Bright green
        self.COLOR_TEXT = (255, 255, 255)

        # --- State Variables (initialized in reset) ---
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.maze_width = 0
        self.maze_height = 0
        self.maze_level = 0
        self.time_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # This will be seeded by the environment's reset method
        self.np_random = None

    def _generate_maze(self, width, height):
        """
        Generates a maze using a recursive backtracking algorithm.
        The maze grid will have dimensions (2*height+1, 2*width+1).
        """
        maze = np.ones((2 * height + 1, 2 * width + 1), dtype=np.uint8)
        
        # Start carving from a random odd-coordinate cell
        start_x, start_y = (
            self.np_random.integers(0, width) * 2 + 1,
            self.np_random.integers(0, height) * 2 + 1,
        )
        
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0

        while stack:
            x, y = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < 2 * width and 0 < ny < 2 * height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # FIX: Correctly select a random neighbor.
                # The original code `self.np_random.choice(len(neighbors), 1)` returned an array
                # which could not be unpacked into `nx, ny`.
                # The correct way is to get a random index and then select the neighbor.
                chosen_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[chosen_index]
                
                # Carve path
                maze[ny, nx] = 0
                maze[(y + ny) // 2, (x + nx) // 2] = 0
                
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure start and exit points are open
        maze[1, 1] = 0
        maze[2 * height - 1, 2 * width - 1] = 0
        
        return maze

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state.
        """
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        # Score is persistent across resets unless it's a new level
        # self.score = 0
        self.game_over = False

        # --- Generate New Maze ---
        maze_dim_cells = min(self.initial_maze_dim + self.maze_level * self.maze_level_increment, self.max_maze_dim)
        self.maze = self._generate_maze(maze_dim_cells, maze_dim_cells)
        self.maze_height, self.maze_width = self.maze.shape
        
        self.player_pos = np.array([1, 1])
        self.exit_pos = np.array([self.maze_width - 2, self.maze_height - 2])
        
        # Set timer based on maze size (Manhattan distance * a factor)
        manhattan_dist = abs(self.exit_pos[0] - self.player_pos[0]) + abs(self.exit_pos[1] - self.player_pos[1])
        self.time_left = int(manhattan_dist * 2.5) # Generous timer

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the environment by one timestep.
        """
        if self.game_over:
            # If the game is already over, return the current state without changes.
            # This is useful for environments that might be stepped into after termination.
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        reward = -0.01  # Small penalty for each step to encourage speed

        # --- Player Movement ---
        new_pos = self.player_pos.copy()
        if movement == 1:  # Up
            new_pos[1] -= 1
        elif movement == 2:  # Down
            new_pos[1] += 1
        elif movement == 3:  # Left
            new_pos[0] -= 1
        elif movement == 4:  # Right
            new_pos[0] += 1

        # --- Collision Detection ---
        # Check bounds and if the new position is a path (0)
        if (0 <= new_pos[0] < self.maze_width and
            0 <= new_pos[1] < self.maze_height and
            self.maze[new_pos[1], new_pos[0]] == 0):
            self.player_pos = new_pos
        else:
            reward -= 0.1 # Penalty for hitting a wall

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if np.array_equal(self.player_pos, self.exit_pos):
            # Win condition
            reward += 10.0
            self.score += 10
            terminated = True
            self.game_over = True
            self.maze_level += 1 # Increase difficulty for next game
        elif self.time_left <= 0:
            # Timeout condition
            reward -= 5.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.max_steps:
            # Step limit reached
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        """
        Renders the current game state to a numpy array.
        """
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), but observation space is (height, width, 3)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        """
        Renders the maze, player, and exit.
        """
        if self.maze is None:
            return

        # --- Calculate rendering dimensions ---
        cell_h = self.screen_height / self.maze_height
        cell_w = self.screen_width / self.maze_width
        cell_size = min(cell_w, cell_h)
        
        maze_render_width = self.maze_width * cell_size
        maze_render_height = self.maze_height * cell_size
        
        offset_x = (self.screen_width - maze_render_width) // 2
        offset_y = (self.screen_height - maze_render_height) // 2

        # --- Render Maze Walls ---
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * cell_size,
                        offset_y + y * cell_size,
                        cell_size + 1, # Add 1 to fill gaps
                        cell_size + 1
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # --- Render Exit ---
        exit_rect = pygame.Rect(
            offset_x + self.exit_pos[0] * cell_size,
            offset_y + self.exit_pos[1] * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # --- Render Player ---
        player_rect = pygame.Rect(
            offset_x + self.player_pos[0] * cell_size,
            offset_y + self.player_pos[1] * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Add a border for better visibility
        pygame.draw.rect(self.screen, (255,255,255), player_rect, 1)


    def _render_ui(self):
        """
        Renders the UI elements like timer and score.
        """
        # --- Render Timer ---
        timer_text = f"TIME: {max(0, self.time_left)}"
        text_surface = self.font.render(timer_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.screen_width - 10, 10))
        
        # Add a dark background for readability
        bg_rect = text_rect.inflate(10, 5)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), bg_rect, border_radius=5)
        
        self.screen.blit(text_surface, text_rect)
        
        # --- Render Score ---
        score_text = f"SCORE: {self.score}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topleft=(10, 10))
        
        bg_rect_score = score_rect.inflate(10, 5)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), bg_rect_score, border_radius=5)
        
        self.screen.blit(score_surface, score_rect)


    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_pos": self.player_pos.tolist(),
            "maze_level": self.maze_level,
        }

    def close(self):
        """
        Cleans up the environment's resources.
        """
        pygame.quit()


if __name__ == "__main__":
    # --- Interactive Play Example ---
    # This block will not run in the headless evaluation
    # but is useful for testing and visualization.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for display
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Maze Runner")
    
    terminated = False
    truncated = False
    
    # --- Main Game Loop ---
    print("\n" + "="*30)
    print("      MAZE RUNNER - MANUAL PLAY")
    print("="*30)
    print(env.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        # Event handling
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
                
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1

                # Only step if a movement key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}, Truncated: {truncated}")

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print("\nGame Over! Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()