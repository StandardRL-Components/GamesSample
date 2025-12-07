
# Generated: 2025-08-27T16:19:08.883001
# Source Brief: brief_00011.md
# Brief Index: 11

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a procedurally generated maze game.
    The player controls a robot that must navigate to an exit within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the robot. Reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated maze to the exit within a limited time. "
        "The maze gets larger and more complex with each success."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()
        
        # Screen and rendering setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants and colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (50, 60, 70)
        self.COLOR_PATH = (30, 35, 40)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_ROBOT_ACCENT = (100, 200, 255)
        self.COLOR_EXIT = (0, 255, 150)
        self.COLOR_EXIT_ACCENT = (150, 255, 200)

        # Game state variables
        self.level = 0
        self.max_steps = 500
        self.maze_dim = 0
        self.maze = None
        self.robot_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state for the first time
        self.reset()

        # Run validation check
        self.validate_implementation()

    def _generate_maze(self, width, height):
        """
        Generates a maze using randomized Depth-First Search (Recursive Backtracker).
        This guarantees a path from any cell to any other cell.
        """
        # 0 = path, 1 = wall
        maze = np.ones((height, width), dtype=np.uint8)
        
        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height

        def carve_path(cx, cy):
            maze[cy, cx] = 0
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and maze[ny, nx] == 1:
                    maze[ny - dy // 2, nx - dx // 2] = 0
                    carve_path(nx, ny)

        # Start carving from a valid path cell (must be odd coordinates)
        carve_path(1, 1)
        
        return maze

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed)
        
        # On win, increase difficulty. On loss/first game, stay at current level.
        if self.game_over and self.robot_pos == self.exit_pos:
            self.level += 1
        
        # Difficulty progression
        self.maze_dim = min(31, 11 + self.level * 2) # Odd numbers for maze generation
        
        # Generate maze and place entities
        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)
        self.robot_pos = (1, 1)
        self.exit_pos = (self.maze_dim - 2, self.maze_dim - 2)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the environment by one time step.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        
        # Default reward for taking a step
        reward = -0.1
        
        # Update robot position based on movement action
        px, py = self.robot_pos
        nx, ny = px, py

        if movement == 1:  # Up
            ny -= 1
        elif movement == 2:  # Down
            ny += 1
        elif movement == 3:  # Left
            nx -= 1
        elif movement == 4:  # Right
            nx += 1
        
        # Check if the new position is valid (not a wall)
        if 0 <= nx < self.maze_dim and 0 <= ny < self.maze_dim and self.maze[ny, nx] == 0:
            self.robot_pos = (nx, ny)

        # Check for termination conditions
        terminated = False
        if self.robot_pos == self.exit_pos:
            reward = 10.0  # Positive reward for reaching the exit
            terminated = True
            # Sound effect placeholder: # play_win_sound()
        elif self.steps >= self.max_steps:
            reward = -10.0 # Negative reward for timeout
            terminated = True
            # Sound effect placeholder: # play_lose_sound()

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
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
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """ Renders the maze, robot, and exit. """
        # Calculate cell size and maze offset for centering
        maze_pixel_size = min(self.width - 40, self.height - 60)
        cell_size = maze_pixel_size / self.maze_dim
        offset_x = (self.width - maze_pixel_size) / 2
        offset_y = (self.height - maze_pixel_size) / 2 + 40 # Offset for UI bar

        # Draw maze paths and walls
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                rect = pygame.Rect(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    math.ceil(cell_size),
                    math.ceil(cell_size)
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Draw Exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            offset_x + ex * cell_size,
            offset_y + ey * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.draw.rect(self.screen, self.COLOR_EXIT_ACCENT, exit_rect.inflate(-cell_size*0.4, -cell_size*0.4))

        # Draw Robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(
            offset_x + rx * cell_size,
            offset_y + ry * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_ACCENT, robot_rect.inflate(-cell_size*0.4, -cell_size*0.4))


    def _render_ui(self):
        """ Renders the UI elements like the timer bar. """
        # Timer bar
        time_ratio = 1.0 - (self.steps / self.max_steps)
        bar_width = self.width - 40
        bar_height = 20
        bar_x = 20
        bar_y = 10

        # Color interpolation from green to yellow to red
        if time_ratio > 0.5:
            # Green to Yellow
            r = int(255 * (1 - time_ratio) * 2)
            g = 255
        else:
            # Yellow to Red
            r = 255
            g = int(255 * time_ratio * 2)
        
        time_color = (r, g, 0)

        # Draw bar background
        pygame.draw.rect(self.screen, self.COLOR_WALL, (bar_x, bar_y, bar_width, bar_height))
        # Draw current time bar
        pygame.draw.rect(self.screen, time_color, (bar_x, bar_y, bar_width * time_ratio, bar_height))
        
        # Draw Level Text
        level_text = self.font.render(f"Level: {self.level + 1}", True, (200, 200, 200))
        self.screen.blit(level_text, (self.width - level_text.get_width() - 20, bar_y))


    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        """
        Closes the environment and cleans up resources.
        """
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.width, env.height))
    
    total_reward = 0
    
    # Game loop
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
                elif event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Game reset. Starting Level {info['level'] + 1}")
                    continue
                
                # Since auto_advance is False, we only step when a key is pressed
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

                if terminated:
                    print("="*20)
                    if info['robot_pos'] == info['exit_pos']:
                        print(f"YOU WIN! Reached exit in {info['steps']} steps.")
                    else:
                        print("GAME OVER! You ran out of time.")
                    print(f"Final Score: {info['score']:.2f}")
                    print("Press 'R' to play again.")
                    print("="*20)
        
        # Render the observation to the display window
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()