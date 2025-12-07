import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based maze puzzle environment. The agent controls a robot that must
    navigate a procedurally generated maze to reach an exit point. The robot
    has a limited number of moves and must avoid various obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one cell at a time. "
        "The goal is to reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated maze to reach the exit. "
        "Avoid obstacles and manage your limited moves to succeed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 38  # Leave some padding
    MAX_STEPS = 20
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_ROBOT = (50, 150, 255)
    COLOR_ROBOT_OUTLINE = (200, 220, 255)
    COLOR_EXIT = (50, 220, 100)
    COLOR_EXIT_GLOW = (150, 255, 180)
    COLOR_PATH = (50, 150, 255, 100)
    OBSTACLE_COLORS = {
        "red": (255, 70, 70),
        "orange": (255, 165, 0),
        "yellow": (255, 220, 50),
    }
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (0, 0, 0, 100)

    # Rewards
    REWARD_EXIT = 100.0
    REWARD_STEP = -0.1
    REWARD_OBSTACLE = {"red": -10.0, "orange": -5.0, "yellow": -2.0}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)


        # Game state variables that persist across resets
        self.successful_episodes = 0
        self.obstacle_density = 0.10  # Start with 10% obstacle density

        # Grid offsets for centering
        self.grid_width = self.GRID_SIZE * self.CELL_SIZE
        self.grid_height = self.GRID_SIZE * self.CELL_SIZE
        self.x_offset = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.y_offset = (self.SCREEN_HEIGHT - self.grid_height) // 2

        # Initialize state variables to None. They will be set in reset().
        self.robot_pos = None
        self.exit_pos = None
        self.obstacles = None
        self.path_trace = None
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize episode-specific state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._generate_maze()
        self.path_trace = [self.robot_pos]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = self.REWARD_STEP
        
        # --- Update game logic ---
        self.steps += 1
        
        # Calculate new position
        prev_pos = self.robot_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if movement != 0:
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            
            # Boundary check
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.robot_pos = new_pos
                self.path_trace.append(new_pos)
                # Sound placeholder: pygame.mixer.Sound("move.wav").play()
        
        # --- Check for terminal conditions ---
        # 1. Reached exit
        if self.robot_pos == self.exit_pos:
            reward += self.REWARD_EXIT
            self.game_over = True
            self.successful_episodes += 1
            # Sound placeholder: pygame.mixer.Sound("win.wav").play()

            # Difficulty scaling
            if self.successful_episodes > 0 and self.successful_episodes % 50 == 0:
                self.obstacle_density = min(0.5, self.obstacle_density + 0.05)

        # 2. Hit an obstacle
        elif self.robot_pos in self.obstacles:
            obstacle_type = self.obstacles[self.robot_pos]
            reward += self.REWARD_OBSTACLE[obstacle_type]
            self.game_over = True
            # Sound placeholder: pygame.mixer.Sound("hit.wav").play()

        # 3. Ran out of moves
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            # Sound placeholder: pygame.mixer.Sound("timeout.wav").play()

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (and transpose for correct shape)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.MAX_STEPS - self.steps,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def _generate_maze(self):
        """Generates a new maze, ensuring the exit is always reachable."""
        while True:
            # Place robot and exit at opposite corners
            self.robot_pos = (0, 0)
            self.exit_pos = (self.GRID_SIZE - 1, self.GRID_SIZE - 1)
            
            # Place obstacles
            self.obstacles = {}
            num_obstacles = int(self.obstacle_density * self.GRID_SIZE * self.GRID_SIZE)
            
            obstacle_types = list(self.OBSTACLE_COLORS.keys())
            
            for _ in range(num_obstacles):
                # Ensure we don't try to place more obstacles than available cells
                if len(self.obstacles) >= self.GRID_SIZE * self.GRID_SIZE - 2:
                    break
                
                while True:
                    pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
                    if pos != self.robot_pos and pos != self.exit_pos and pos not in self.obstacles:
                        obstacle_type = self.np_random.choice(obstacle_types)
                        self.obstacles[pos] = obstacle_type
                        break
            
            # Validate that a path exists
            if self._is_path_available():
                break

    def _is_path_available(self):
        """Uses Breadth-First Search (BFS) to check for a path."""
        q = deque([self.robot_pos])
        visited = {self.robot_pos}
        
        while q:
            current_pos = q.popleft()
            
            if current_pos == self.exit_pos:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if (0 <= next_pos[0] < self.GRID_SIZE and
                    0 <= next_pos[1] < self.GRID_SIZE and
                    next_pos not in self.obstacles and
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    q.append(next_pos)
        return False

    def _grid_to_pixels(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for drawing."""
        gx, gy = grid_pos
        px = self.x_offset + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.y_offset + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.x_offset + i * self.CELL_SIZE, self.y_offset),
                             (self.x_offset + i * self.CELL_SIZE, self.y_offset + self.grid_height), 1)
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.x_offset, self.y_offset + i * self.CELL_SIZE),
                             (self.x_offset + self.grid_width, self.y_offset + i * self.CELL_SIZE), 1)

        # Draw path trace
        if self.path_trace and len(self.path_trace) > 1:
            pixel_path = [self._grid_to_pixels(p) for p in self.path_trace]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, pixel_path, 3)
            
        # Draw obstacles
        if self.obstacles:
            for pos, o_type in self.obstacles.items():
                px, py = self._grid_to_pixels(pos)
                rect = pygame.Rect(px - self.CELL_SIZE // 2 + 3, py - self.CELL_SIZE // 2 + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
                pygame.draw.rect(self.screen, self.OBSTACLE_COLORS[o_type], rect, border_radius=4)
        
        # Draw exit
        if self.exit_pos:
            ex, ey = self._grid_to_pixels(self.exit_pos)
            exit_rect = pygame.Rect(ex - self.CELL_SIZE // 2 + 2, ey - self.CELL_SIZE // 2 + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            # Glow effect
            glow_rect = pygame.Rect(ex - self.CELL_SIZE // 2, ey - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_EXIT_GLOW, glow_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=6)

        # Draw robot
        if self.robot_pos:
            rx, ry = self._grid_to_pixels(self.robot_pos)
            robot_rect = pygame.Rect(rx - self.CELL_SIZE // 2 + 4, ry - self.CELL_SIZE // 2 + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT_OUTLINE, robot_rect, 2, border_radius=4)

    def _render_ui(self):
        # Semi-transparent background for UI
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))
        
        # Moves left
        moves_text = self.font_large.render(f"Moves: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 7))

        # Score
        score_str = f"Score: {self.score:.1f}"
        score_text = self.font_large.render(score_str, True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(right=self.SCREEN_WIDTH - 15, top=7)
        self.screen.blit(score_text, score_rect)

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.robot_pos == self.exit_pos:
                msg = "SUCCESS!"
                color = self.COLOR_EXIT
            else:
                msg = "FAILURE"
                color = self.OBSTACLE_COLORS["red"]

            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text, text_rect)
            self.screen.blit(overlay, (0, 0))


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space by calling reset() first.
        # This initializes the game state, which is required before get_observation()
        # or step() can be called.
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Manual testing not available in headless mode. Exiting.")
        exit()
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Maze Robot")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    while running:
        action = [0, 0, 0]  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if terminated:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    continue

                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Only step if a movement key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)  # Limit frame rate
        
    env.close()