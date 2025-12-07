import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to navigate the maze. Find the green exit before the steps run out."
    )

    # User-facing game description
    game_description = (
        "Navigate a procedurally generated maze to find the exit within a limited number of steps. "
        "Larger mazes appear every 3 levels. Entering a dead-end incurs a score penalty."
    )

    # Frames advance on action, not automatically
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = Box(
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

        # --- Visuals ---
        self.COLOR_BG = pygame.Color("#1A2A3A")
        self.COLOR_WALL = pygame.Color("#3A506B")
        self.COLOR_PATH = pygame.Color("#0B132B")
        self.COLOR_PLAYER = pygame.Color("#FFD700")
        self.COLOR_PLAYER_GLOW = pygame.Color(255, 215, 0, 50)
        self.COLOR_EXIT = pygame.Color("#4CFFB1")
        self.COLOR_TEXT = pygame.Color("#FFFFFF")
        self.COLOR_SHADOW = pygame.Color("#000000")
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # --- Game State ---
        self.level = 0
        self.completed_in_a_row = 0
        self.maze_dim = 10
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.dead_ends = set()
        self.steps = 0
        self.steps_remaining = 0
        self.score = 0
        self.game_over = False

        # --- Game Parameters ---
        self.MAX_EPISODE_STEPS = 1000
        self.REWARD_EXIT = 100.0
        self.REWARD_DEAD_END = -1.0
        self.REWARD_STEP = -0.1

        # Initialize state variables
        # The first call to reset() needs a seed to be reproducible
        self.reset(seed=42)

    def _generate_maze(self, width, height):
        # Use numpy for efficient grid manipulation
        # 1 for wall, 0 for path
        maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=np.uint8)
        
        # Start carving from a random odd position
        start_x, start_y = self.np_random.integers(0, width) * 2 + 1, self.np_random.integers(0, height) * 2 + 1
        maze[start_y, start_x] = 0
        
        stack = [(start_x, start_y)]
        
        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width * 2 and 0 < ny < height * 2 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # FIX: Correctly select and unpack a random neighbor
                # 1. Get a random index for the neighbors list.
                # 2. Use the index to get the (x, y) tuple.
                neighbor_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[neighbor_index]
                
                # Carve path to neighbor
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
                
        return maze

    def _identify_dead_ends(self, maze):
        dead_ends = set()
        height, width = maze.shape
        for r in range(1, height, 2):
            for c in range(1, width, 2):
                if maze[r, c] == 0: # It's a path cell
                    path_neighbors = 0
                    if maze[r-1, c] == 0: path_neighbors += 1 # Up
                    if maze[r+1, c] == 0: path_neighbors += 1 # Down
                    if maze[r, c-1] == 0: path_neighbors += 1 # Left
                    if maze[r, c+1] == 0: path_neighbors += 1 # Right
                    
                    # A dead end has only one exit, excluding start/end points
                    if path_neighbors == 1:
                        logical_pos = ((c - 1) // 2, (r - 1) // 2)
                        if logical_pos != self.player_pos and logical_pos != self.exit_pos:
                            dead_ends.add(logical_pos)
        return dead_ends

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Difficulty progression
        if self.game_over and self.player_pos == self.exit_pos: # Won last game
            self.completed_in_a_row += 1
            if self.completed_in_a_row % 3 == 0:
                self.level += 1
        elif self.game_over: # Lost last game
            self.completed_in_a_row = 0

        self.maze_dim = 10 + self.level * 2
        
        # Generate maze and features
        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)
        self.player_pos = (0, 0)
        self.exit_pos = (self.maze_dim - 1, self.maze_dim - 1)
        self.dead_ends = self._identify_dead_ends(self.maze)

        # Reset state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.steps_remaining = self.maze_dim * self.maze_dim # Scaled time limit

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        reward = self.REWARD_STEP

        px, py = self.player_pos
        nx, ny = px, py

        # Map action to grid movement
        if movement == 1: ny -= 1 # Up
        elif movement == 2: ny += 1 # Down
        elif movement == 3: nx -= 1 # Left
        elif movement == 4: nx += 1 # Right
        
        # Check for valid move
        if 0 <= nx < self.maze_dim and 0 <= ny < self.maze_dim:
            # Convert logical coords to maze array coords
            maze_x, maze_y = px * 2 + 1, py * 2 + 1
            n_maze_x, n_maze_y = nx * 2 + 1, ny * 2 + 1
            wall_x, wall_y = (maze_x + n_maze_x) // 2, (maze_y + n_maze_y) // 2
            
            if self.maze[wall_y, wall_x] == 0: # Path is clear
                self.player_pos = (nx, ny)

        # Check for events at the new position
        if self.player_pos in self.dead_ends:
            reward += self.REWARD_DEAD_END
            self.dead_ends.remove(self.player_pos) # Only penalize once

        terminated = False
        if self.player_pos == self.exit_pos:
            reward += self.REWARD_EXIT
            terminated = True

        # Update counters
        self.steps += 1
        self.steps_remaining -= 1
        self.score += reward

        # Check termination conditions
        if self.steps_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            
        self.game_over = terminated

        # The 'truncated' flag is generally used for time-limits, while 'terminated' is for terminal states.
        # Here, both running out of steps and reaching the exit are terminal states.
        truncated = False 

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate maze rendering properties
        padding = 50
        maze_render_area_size = min(self.screen_width - padding, self.screen_height - padding)
        cell_size = maze_render_area_size / self.maze_dim
        offset_x = (self.screen_width - self.maze_dim * cell_size) / 2
        offset_y = (self.screen_height - self.maze_dim * cell_size) / 2

        # Draw Maze Walls
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                if self.maze[r, c] == 1:
                    # Walls are between cells, so their size is different
                    if r % 2 == 0 and c % 2 == 1: # Horizontal wall
                        rx = offset_x + ((c - 1) // 2) * cell_size
                        ry = offset_y + (r // 2) * cell_size
                        pygame.draw.line(self.screen, self.COLOR_WALL, (rx, ry), (rx + cell_size, ry), 3)
                    elif r % 2 == 1 and c % 2 == 0: # Vertical wall
                        rx = offset_x + (c // 2) * cell_size
                        ry = offset_y + ((r - 1) // 2) * cell_size
                        pygame.draw.line(self.screen, self.COLOR_WALL, (rx, ry), (rx, ry + cell_size), 3)

        # Draw Exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            offset_x + ex * cell_size,
            offset_y + ey * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw Player
        px, py = self.player_pos
        player_center_x = int(offset_x + px * cell_size + cell_size / 2)
        player_center_y = int(offset_y + py * cell_size + cell_size / 2)
        player_radius = int(cell_size * 0.35)
        
        # Glow effect
        glow_radius = int(player_radius * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_center_x - glow_radius, player_center_y - glow_radius))
        
        # Player circle
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_center_x, player_center_y), player_radius)


    def _render_ui(self):
        def draw_text(text, font, color, x, y, align="topleft"):
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            shadow_surface = font.render(text, True, self.COLOR_SHADOW)
            if align == "topleft":
                text_rect.topleft = (x, y)
            elif align == "topright":
                text_rect.topright = (x, y)
            elif align == "midtop":
                text_rect.midtop = (x, y)
            self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surface, text_rect)

        # Score
        draw_text(f"SCORE: {int(self.score)}", self.font_large, self.COLOR_TEXT, 20, 10, "topleft")

        # Steps Remaining
        draw_text(f"STEPS: {self.steps_remaining}", self.font_large, self.COLOR_TEXT, self.screen_width - 20, 10, "topright")
        
        # Level
        draw_text(f"LEVEL: {self.level + 1}", self.font_large, self.COLOR_TEXT, self.screen_width // 2, 10, "midtop")


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level + 1,
            "steps_remaining": self.steps_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display driver for manual play
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' on Windows, 'x11' or 'wayland' on Linux

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Re-initialize Pygame with a display for manual play
    pygame.display.init()
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    
    print(env.user_guide)

    running = True
    while running:
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
                else:
                    action[0] = 0 # No-op for other keys
                
                obs, reward, terminated, truncated, info = env.step(action)
                action[0] = 0 # Reset action to no-op after one step
                
                print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}")

                if terminated:
                    print("Game Over!")
                    # Render the final frame
                    frame = env._get_observation()
                    frame = np.transpose(frame, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    # Wait a bit before resetting
                    pygame.time.wait(2000)
                    obs, info = env.reset()
                    terminated = False


        # Render the observation to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for manual play

    env.close()