import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze to find the exit. Each win increases the maze size, making the next challenge harder. Manage your time wisely!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.screen_width = 640
        self.screen_height = 400
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 20)
            self.font_game_over = pygame.font.SysFont("arial", 48)

        # Game constants and colors
        self.max_steps = 600  # 60 seconds at 10 steps/sec
        self.initial_maze_dim = 10
        self.max_maze_dim = 40 # Capped for visual clarity on screen

        self.COLOR_BG = (25, 25, 40)
        self.COLOR_WALL = (70, 70, 90)
        self.COLOR_PATH = self.COLOR_BG
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (50, 150, 255, 50)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_EXIT_GLOW = (50, 255, 150, 60)
        self.COLOR_TEXT = (240, 240, 240)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.maze_width = self.initial_maze_dim
        self.maze_height = self.initial_maze_dim
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        
    def _generate_maze(self, width, height):
        # Maze dimensions must be odd for the algorithm
        grid_w, grid_h = (width * 2 + 1, height * 2 + 1)
        maze = np.ones((grid_h, grid_w), dtype=np.uint8) # 1 = wall
        
        # Use numpy's random generator
        start_x, start_y = (self.np_random.integers(0, width) * 2 + 1, 
                            self.np_random.integers(0, height) * 2 + 1)
        
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0 # 0 = path

        while stack:
            current_x, current_y = stack[-1]
            neighbors = []
            
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = current_x + dx, current_y + dy
                if 0 < nx < grid_w and 0 < ny < grid_h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # FIX: Correctly select and unpack a random neighbor
                random_index = self.np_random.integers(len(neighbors))
                next_x, next_y = neighbors[random_index]
                
                # Carve path to neighbor
                maze[next_y, next_x] = 0
                maze[current_y + (next_y - current_y) // 2, current_x + (next_x - current_x) // 2] = 0
                
                stack.append((next_x, next_y))
            else:
                stack.pop()
        
        # Select start and exit positions from path cells
        path_cells = np.argwhere(maze == 0)
        
        # Ensure start and exit are far apart
        while True:
            start_idx, exit_idx = self.np_random.choice(len(path_cells), 2, replace=False)
            player_start = tuple(path_cells[start_idx][::-1]) # (x, y)
            exit_pos = tuple(path_cells[exit_idx][::-1])
            
            manhattan_dist = abs(player_start[0] - exit_pos[0]) + abs(player_start[1] - exit_pos[1])
            if manhattan_dist > (width + height) * 0.75: # Ensure they are reasonably far
                break

        return maze, player_start, exit_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # On the very first reset, or after a timeout, reset difficulty
        if not hasattr(self, 'last_win') or not self.last_win:
            self.maze_width = self.initial_maze_dim
            self.maze_height = self.initial_maze_dim

        self.maze, self.player_pos, self.exit_pos = self._generate_maze(self.maze_width, self.maze_height)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_win = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1  # Small penalty for each step taken
        
        px, py = self.player_pos
        new_px, new_py = px, py

        if movement == 1:  # Up
            new_py -= 1
        elif movement == 2:  # Down
            new_py += 1
        elif movement == 3:  # Left
            new_px -= 1
        elif movement == 4:  # Right
            new_px += 1
        
        # Check for wall collision
        if self.maze[new_py, new_px] == 1:
            reward -= 1.0  # Penalty for hitting a wall
            # Player does not move
        elif movement != 0:
            self.player_pos = (new_px, new_py)

        self.steps += 1
        terminated = False

        # Check for win condition
        if self.player_pos == self.exit_pos:
            time_bonus = max(0, (self.max_steps - self.steps) * 0.1)
            win_reward = 50.0 + time_bonus
            reward += win_reward
            terminated = True
            self.game_over = True
            self.last_win = True
            
            # Increase difficulty for next round
            self.maze_width = min(self.max_maze_dim, int(self.maze_width * 1.1) + 1)
            self.maze_height = min(self.max_maze_dim, int(self.maze_height * 1.1) + 1)

        # Check for loss condition (timeout)
        elif self.steps >= self.max_steps:
            timeout_penalty = -100.0
            reward += timeout_penalty
            terminated = True
            self.game_over = True
            self.last_win = False # Failed, so difficulty will reset

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        grid_h, grid_w = self.maze.shape
        
        # Calculate tile size to fit the maze on screen
        tile_w = self.screen_width / grid_w
        tile_h = self.screen_height / grid_h
        
        # Render maze walls
        for y in range(grid_h):
            for x in range(grid_w):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * tile_w, y * tile_h, math.ceil(tile_w), math.ceil(tile_h)))

        # Visual effect for exit
        exit_pulse = 0.6 + 0.4 * math.sin(self.steps * 0.2)
        ex, ey = self.exit_pos
        exit_rect_inner = pygame.Rect(ex * tile_w, ey * tile_h, tile_w, tile_h)
        pygame.gfxdraw.box(self.screen, exit_rect_inner, self.COLOR_EXIT)
        # Glow effect
        glow_radius = int(max(tile_w, tile_h) * exit_pulse)
        pygame.gfxdraw.filled_circle(self.screen, int(exit_rect_inner.centerx), int(exit_rect_inner.centery), glow_radius, self.COLOR_EXIT_GLOW)


        # Render player
        player_pulse = 0.8 + 0.2 * math.sin(self.steps * 0.3)
        px, py = self.player_pos
        player_rect_inner = pygame.Rect(px * tile_w + tile_w * 0.1, py * tile_h + tile_h * 0.1, tile_w * 0.8, tile_h * 0.8)
        
        # Glow effect
        glow_radius = int(max(tile_w, tile_h) * 0.6 * player_pulse)
        pygame.gfxdraw.filled_circle(self.screen, int(player_rect_inner.centerx), int(player_rect_inner.centery), glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player square
        pygame.gfxdraw.box(self.screen, player_rect_inner, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score display
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Timer display
        time_left = max(0.0, (self.max_steps - self.steps) / 10.0)
        timer_text = f"Time: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        timer_rect = timer_surf.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(timer_surf, timer_rect)

        # Game over message
        if self.game_over:
            if self.last_win:
                msg = "YOU WIN!"
            else:
                msg = "TIME OUT"
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            game_over_rect = game_over_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(game_over_surf, game_over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "maze_size": (self.maze_width, self.maze_height),
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # To play, you might need to comment out the SDL_VIDEODRIVER line
    # and install pygame: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a dummy screen for display if running as main
    pygame.display.set_caption("Maze Runner")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    terminated = False
    running = True
    total_reward = 0.0
    
    # Game loop
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
            # If terminated, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                terminated = False
                total_reward = 0.0

        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need to control the speed for human play
        env.clock.tick(15) 

    env.close()