
# Generated: 2025-08-28T02:05:31.579929
# Source Brief: brief_01595.md
# Brief Index: 1595

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a grid-based robot navigation puzzle.
    The agent controls a robot on a 10x10 grid and must navigate it to a
    goal location while avoiding obstacles. Each episode features a new,
    procedurally generated, and guaranteed-solvable maze.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: ↑↓←→ to move the robot. Reach the green goal."

    # User-facing game description
    game_description = "Navigate a robot through obstacle-laden grids to reach a goal before the step limit."

    # The game is turn-based; it only advances on an action.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    GRID_DIM = 400  # Pixel dimension of the square grid
    CELL_SIZE = GRID_DIM // GRID_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_DIM) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_DIM) // 2

    # Visual style
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 70)
    COLOR_ROBOT = (255, 80, 80)
    COLOR_GOAL = (80, 255, 80)
    COLOR_OBSTACLE = (80, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TITLE = (180, 180, 200)

    # Gameplay parameters
    MAX_STEPS = 500
    INITIAL_OBSTACLES = 3
    MAX_OBSTACLES = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 24, bold=True)

        # Game state variables
        self.robot_pos = [0, 0]
        self.goal_pos = [0, 0]
        self.obstacle_pos = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Difficulty progression
        self.successful_episodes = 0
        self.num_obstacles = self.INITIAL_OBSTACLES

        # This will be initialized in reset()
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Update difficulty every 5 successful episodes
        self.num_obstacles = min(
            self.MAX_OBSTACLES,
            self.INITIAL_OBSTACLES + (self.successful_episodes // 5)
        )
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Default penalty for taking a step
        terminated = False
        
        # --- Update robot position ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if movement != 0:
            next_x = self.robot_pos[0] + dx
            next_y = self.robot_pos[1] + dy

            # Check grid boundaries
            if 0 <= next_x < self.GRID_SIZE and 0 <= next_y < self.GRID_SIZE:
                self.robot_pos = [next_x, next_y]

        # --- Check for game events ---
        if self.robot_pos == self.goal_pos:
            reward = 100.0
            terminated = True
            self.successful_episodes += 1
            # Sound: Goal reached success chime
        elif self.robot_pos in self.obstacle_pos:
            reward = -10.0
            terminated = True
            # Sound: Collision/failure buzz
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True # Episode ends due to step limit

        self.score += reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _generate_level(self):
        """
        Generates a new level layout, guaranteeing a solvable path from
        the robot to the goal.
        """
        while True:
            # Get all possible grid coordinates
            all_pos = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            self.np_random.shuffle(all_pos)

            # Assign unique positions for robot, goal, and obstacles
            self.robot_pos = list(all_pos.pop())
            self.goal_pos = list(all_pos.pop())
            self.obstacle_pos = [list(pos) for pos in all_pos[:self.num_obstacles]]

            # Verify a path exists; if not, regenerate the level
            if self._is_path_available():
                break

    def _is_path_available(self):
        """
        Performs a Breadth-First Search (BFS) to check if a path exists
        from the robot's start position to the goal, avoiding obstacles.
        """
        q = deque([tuple(self.robot_pos)])
        visited = {tuple(self.robot_pos)}
        obstacle_set = {tuple(pos) for pos in self.obstacle_pos}

        while q:
            x, y = q.popleft()

            if (x, y) == tuple(self.goal_pos):
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and
                        (nx, ny) not in visited and (nx, ny) not in obstacle_set):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _get_observation(self):
        """Renders the current game state to a Pygame surface and returns it as a NumPy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_entities()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        """Draws the grid lines."""
        for x in range(self.GRID_SIZE + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_DIM), 1)
        for y in range(self.GRID_SIZE + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_DIM, py), 1)

    def _render_entities(self):
        """Draws the robot, goal, and obstacles on the grid."""
        # Obstacles
        for ox, oy in self.obstacle_pos:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + ox * self.CELL_SIZE,
                self.GRID_OFFSET_Y + oy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect.inflate(-4, -4))

        # Goal
        gx, gy = self.goal_pos
        goal_rect = pygame.Rect(
            self.GRID_OFFSET_X + gx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + gy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect.inflate(-4, -4))

        # Robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(
            self.GRID_OFFSET_X + rx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + ry * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect.inflate(-4, -4))

    def _render_ui(self):
        """Renders the UI elements like score and step count."""
        title_surf = self.font_title.render("GRIDBOT", True, self.COLOR_TITLE)
        self.screen.blit(title_surf, (20, 20))

        score_text = f"Score: {self.score:.1f}"
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        level_text = f"Level: {self.successful_episodes // 5 + 1}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        level_surf = self.font_ui.render(level_text, True, self.COLOR_TEXT)

        self.screen.blit(score_surf, (20, 80))
        self.screen.blit(steps_surf, (20, 105))
        self.screen.blit(level_surf, (20, 130))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "successful_episodes": self.successful_episodes,
        }
    
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be executed when the environment is used by Gymnasium runners
    
    # Set SDL to a dummy driver to run without a display
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- For visual testing, re-enable the display ---
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gridbot")
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Game Test ---")
    print(GameEnv.user_guide)
    
    # Game loop
    running = True
    while running:
        # Get observation from the environment and display it
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
            print("Press any key to restart.")
            
            wait_for_key = True
            while wait_for_key:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_key = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        wait_for_key = False
            
            if running:
                obs, info = env.reset()
                terminated = False
                total_reward = 0
            continue

        # Map keyboard inputs to actions
        action = [0, 0, 0] # Default action: no-op
        
        event_processed = False
        while not event_processed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    event_processed = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    
                    event_processed = True # Process one key press per frame
            
            # If no key was pressed, we still need to exit the loop
            if not pygame.event.peek(pygame.KEYDOWN):
                 event_processed = True

        if not running:
            break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    pygame.quit()
    print("Game exited.")