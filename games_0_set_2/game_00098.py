import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A Gymnasium environment for a maze navigation game.

    The player must navigate a procedurally generated maze to find the exit
    within a given time limit. Collecting bonus items grants extra time.
    The maze complexity increases with each successfully completed level.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game state.

    **Rewards:**
    - -0.01 per step (encourages speed).
    - +10 for collecting a bonus item.
    - +100 for reaching the exit.
    - -100 for running out of time.

    **Termination:**
    - The episode ends if the player reaches the exit.
    - The episode ends if the timer reaches zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate procedurally generated mazes to reach the exit within the time limit. Collect gold stars for a time bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1800  # 60 seconds at 30fps (if auto_advance=True), but here it's a step limit.
        self.INITIAL_TIME = 600 # 60 seconds worth of steps
        self.BONUS_TIME = 150 # 15 seconds bonus

        # --- Colors and Fonts ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (70, 80, 100)
        self.COLOR_PATH = self.COLOR_BG
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_BONUS = (255, 223, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TIMER_WARN = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        # Set a dummy video driver for headless execution
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.time_remaining = 0
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.bonus_pos = (0, 0)
        self.bonus_active = False
        self.maze = np.array([[]])
        self.maze_dims = (0, 0)
        self.won_last_game = False
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game Logic ---
        if self.won_last_game:
            self.level += 1
        else:
            self.level = 1
        self.won_last_game = False

        self.steps = 0
        self.score = 0
        self.time_remaining = self.INITIAL_TIME

        # --- Maze Generation ---
        maze_w = min(30, math.ceil(10 * (1.1 ** (self.level - 1))))
        maze_h = min(22, math.ceil(7 * (1.1 ** (self.level - 1))))
        self.maze_dims = (maze_w, maze_h)
        self.maze = self._generate_maze(maze_w, maze_h)

        # --- Entity Placement ---
        self.player_pos = (1, 1)
        self.exit_pos = (maze_w * 2 - 1, maze_h * 2 - 1)

        # Ensure the exit is a path tile, in case the generator didn't reach it.
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 0

        # Place bonus item randomly on a path tile, avoiding player and exit
        path_tiles = np.argwhere(self.maze == 0).tolist()

        # Create [y, x] representations for comparison, matching argwhere's output
        player_pos_yx = [self.player_pos[1], self.player_pos[0]]
        exit_pos_yx = [self.exit_pos[1], self.exit_pos[0]]

        # Use a list comprehension for safe removal of player and exit positions
        candidate_tiles = [
            tile for tile in path_tiles
            if tile != player_pos_yx and tile != exit_pos_yx
        ]

        if candidate_tiles:
            # Select a random tile from the candidates by index for clarity and safety
            idx = self.np_random.integers(0, len(candidate_tiles))
            bonus_pos_yx = candidate_tiles[idx]
            # Convert chosen [y, x] back to (x, y) for game logic
            self.bonus_pos = (bonus_pos_yx[1], bonus_pos_yx[0])
            self.bonus_active = True
        else:
            self.bonus_active = False
            
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False

        # --- Player Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            # Player moves 2 cells in maze grid, checks wall in between
            wall_check_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            new_pos = (self.player_pos[0] + dx * 2, self.player_pos[1] + dy * 2)

            if 0 <= new_pos[0] < self.maze.shape[1] and 0 <= new_pos[1] < self.maze.shape[0]:
                if self.maze[wall_check_pos[1], wall_check_pos[0]] == 0:
                    self.player_pos = new_pos

        # --- Update Game State ---
        self.steps += 1
        self.time_remaining -= 1

        # --- Check Game Events ---
        # Bonus collection
        if self.bonus_active and self.player_pos == self.bonus_pos:
            reward += 10.0
            self.time_remaining = min(self.INITIAL_TIME, self.time_remaining + self.BONUS_TIME)
            self.bonus_active = False
            self._create_particles(self.bonus_pos, self.COLOR_BONUS)
            # sfx: bonus_collect.wav

        # Reached exit
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            self.won_last_game = True
            # sfx: level_complete.wav

        # Timeout
        if self.time_remaining <= 0:
            if not terminated: # Don't penalize if won on the last step
                reward -= 100.0
            terminated = True
            # sfx: game_over.wav

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "level": self.level,
            "time_remaining": self.time_remaining,
        }

    def _generate_maze(self, width, height):
        # Maze is represented by a grid where (2w+1, 2h+1) allows for walls between cells
        grid_w, grid_h = width * 2 + 1, height * 2 + 1
        maze = np.ones((grid_h, grid_w), dtype=np.uint8)
        stack = []
        
        # Start DFS from cell (1,1)
        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < grid_w and 0 < ny < grid_h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(np.array(neighbors), axis=0)
                # Carve path
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def _render_game(self):
        # --- Calculate rendering metrics ---
        maze_pixel_w = self.maze.shape[1] * 10
        maze_pixel_h = self.maze.shape[0] * 10
        
        # Fit maze to screen while maintaining aspect ratio
        scale_w = (self.WIDTH - 40) / self.maze.shape[1]
        scale_h = (self.HEIGHT - 80) / self.maze.shape[0]
        cell_size = math.floor(min(scale_w, scale_h))
        
        render_w = self.maze.shape[1] * cell_size
        render_h = self.maze.shape[0] * cell_size
        offset_x = (self.WIDTH - render_w) // 2
        offset_y = (self.HEIGHT - render_h) // 2 + 40 # Push down for UI

        # --- Draw Maze ---
        for y in range(self.maze.shape[0]):
            for x in range(self.maze.shape[1]):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * cell_size,
                        offset_y + y * cell_size,
                        cell_size,
                        cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # --- Draw Exit ---
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(offset_x + ex * cell_size, offset_y + ey * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        
        # --- Draw Bonus ---
        if self.bonus_active:
            bx, by = self.bonus_pos
            self._draw_star(
                self.screen, self.COLOR_BONUS,
                (offset_x + int((bx + 0.5) * cell_size), offset_y + int((by + 0.5) * cell_size)),
                int(cell_size * 0.4)
            )

        # --- Draw Particles ---
        self._update_and_draw_particles(offset_x, offset_y, cell_size)

        # --- Draw Player ---
        px, py = self.player_pos
        player_rect = pygame.Rect(offset_x + px * cell_size, offset_y + py * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Add a border for visibility
        pygame.draw.rect(self.screen, tuple(min(255, c+60) for c in self.COLOR_PLAYER), player_rect, max(1, int(cell_size/8)))

    def _render_ui(self):
        # --- Level Display ---
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 15))

        # --- Timer Display ---
        time_str = f"Time: {max(0, self.time_remaining / 10):.1f}"
        timer_color = self.COLOR_TEXT if self.time_remaining > 100 else self.COLOR_TIMER_WARN
        time_text = self.font_large.render(time_str, True, timer_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_text, time_rect)

    def _draw_star(self, surface, color, center, radius):
        points = []
        for i in range(10):
            angle = math.radians(i * 36)
            r = radius if i % 2 == 0 else radius * 0.4
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _create_particles(self, grid_pos, color):
        for _ in range(20):
            self.particles.append({
                "pos": list(grid_pos), # grid coordinates
                "vel": [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)],
                "life": self.np_random.integers(15, 30),
                "color": color,
            })
    
    def _update_and_draw_particles(self, offset_x, offset_y, cell_size):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                # Fade out effect
                alpha = max(0, min(255, int(255 * (p["life"] / 20))))
                color = (*p["color"], alpha)
                
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                
                # Draw particle centered in its "cell"
                center_x = int((p["pos"][0] + 0.5) * cell_size) + offset_x
                center_y = int((p["pos"][1] + 0.5) * cell_size) + offset_y
                
                pygame.draw.circle(
                    self.screen,
                    p["color"],
                    (center_x, center_y),
                    max(1, int(p["life"] / 10 * cell_size * 0.1))
                )

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

# Example of how to run the environment
if __name__ == '__main__':
    # The environment is designed for headless execution, so we set the SDL_VIDEODRIVER
    # to "dummy" to prevent Pygame from trying to open a window.
    # This is done in the __init__ method.

    env = GameEnv()
    
    # --- Human Play Example ---
    # This part requires a display. Comment out the os.environ line in __init__ to run.
    # To run this, you need to have a display available.
    # try:
    #     os.environ.pop("SDL_VIDEODRIVER")
    # except KeyError:
    #     pass
    #
    # pygame.display.init()
    # pygame.font.init()
    # env_human = GameEnv(render_mode="human") # 'human' mode is not defined, but we can fake it
    # screen = pygame.display.set_mode((env_human.WIDTH, env_human.HEIGHT))
    # pygame.display.set_caption("Maze Runner")
    #
    # obs, info = env_human.reset()
    # terminated = False
    # clock = pygame.time.Clock()
    # while not terminated:
    #     action = [0, 0, 0] # Default action: no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             terminated = True
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
    #
    #     obs, reward, terminated, truncated, info = env_human.step(action)
    #
    #     # Blit the observation to the display screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
    #
    #     if terminated:
    #         print("Game Over! Resetting...")
    #         pygame.time.wait(2000)
    #         obs, info = env_human.reset()
    #         terminated = False
    #
    #     clock.tick(10) # Limit FPS for human playability
    #
    # env_human.close()

    # --- Agent Interaction Example ---
    obs, info = env.reset()
    print("Initial state:")
    print(f"Info: {info}")

    total_reward = 0
    for i in range(200):
        action = env.action_space.sample()  # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 50 == 0:
            print(f"\n--- Step {i+1} ---")
            print(f"Action taken: {action}")
            print(f"Info: {info}")
            print(f"Reward this step: {reward:.4f}")
            print(f"Total reward so far: {total_reward:.4f}")

        if terminated:
            print(f"\nEpisode finished after {i+1} steps.")
            print(f"Final Info: {info}")
            print(f"Final Score: {info['score']:.4f}")
            break
    
    env.close()