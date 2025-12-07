import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


# Set a dummy video driver to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys (↑↓←→) to navigate the maze."
    game_description = "Navigate procedurally generated mazes against the clock to reach the exit."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 120, 180)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 40)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TEXT = (230, 230, 230)
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State ---
        self.level = 1
        self.base_maze_dims = (20, 12)  # w, h to match screen aspect ratio
        self.just_won = False
        
        # Initialize state variables - will be properly set in reset()
        self.maze_dims = (0, 0)
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.start_time = 60.0
        self.time_remaining = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False

        # This call to reset() is needed to initialize the maze before validation
        # We need a seed for the first call to reset
        self.reset(seed=0)
        
        # --- Validation ---
        # self.validate_implementation() # This is better called outside __init__
    
    def _generate_maze(self, width, height):
        # Maze represented as a grid of cells. Each cell is a dict of its walls.
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(width)] for _ in range(height)]
        
        # Randomized DFS to carve paths
        stack = []
        start_cell = (0, 0)
        visited = {start_cell}
        stack.append(start_cell)

        while stack:
            r, c = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            if r > 0 and (r - 1, c) not in visited: neighbors.append(('N', (r - 1, c)))
            if r < height - 1 and (r + 1, c) not in visited: neighbors.append(('S', (r + 1, c)))
            if c > 0 and (r, c - 1) not in visited: neighbors.append(('W', (r, c - 1)))
            if c < width - 1 and (r, c + 1) not in visited: neighbors.append(('E', (r, c + 1)))

            if neighbors:
                # Choose a random neighbor
                # FIX: np.random.choice tries to convert the list of tuples to a numpy array, causing a ValueError.
                # Instead, we should pick a random index from the list.
                neighbor_index = self.np_random.integers(len(neighbors))
                direction, (next_r, next_c) = neighbors[neighbor_index]
                
                # Knock down walls between current and next cell
                if direction == 'N':
                    maze[r][c]['N'] = False
                    maze[next_r][next_c]['S'] = False
                elif direction == 'S':
                    maze[r][c]['S'] = False
                    maze[next_r][next_c]['N'] = False
                elif direction == 'W':
                    maze[r][c]['W'] = False
                    maze[next_r][next_c]['E'] = False
                elif direction == 'E':
                    maze[r][c]['E'] = False
                    maze[next_r][next_c]['W'] = False
                
                visited.add((next_r, next_c))
                stack.append((next_r, next_c))
            else:
                # Backtrack
                stack.pop()
        
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.just_won:
            self.level += 1
        self.just_won = False

        self.maze_dims = (
            int(self.base_maze_dims[0] * (1 + (self.level - 1) * 0.1)),
            int(self.base_maze_dims[1] * (1 + (self.level - 1) * 0.1))
        )
        
        self.maze = self._generate_maze(self.maze_dims[0], self.maze_dims[1])
        self.player_pos = (0, 0) # row, col
        self.exit_pos = (self.maze_dims[1] - 1, self.maze_dims[0] - 1)

        self.start_time = 60.0
        self.time_remaining = self.start_time
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._handle_movement(movement)
        
        self.time_remaining = max(0, self.time_remaining - 1.0 / 30.0)
        self.steps += 1
        
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        self.game_over = terminated
        
        truncated = self.steps >= 1800
        if truncated:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        r, c = self.player_pos
        if movement == 1 and r > 0 and not self.maze[r][c]['N']: # Up
            self.player_pos = (r - 1, c)
        elif movement == 2 and r < self.maze_dims[1] - 1 and not self.maze[r][c]['S']: # Down
            self.player_pos = (r + 1, c)
        elif movement == 3 and c > 0 and not self.maze[r][c]['W']: # Left
            self.player_pos = (r, c - 1)
        elif movement == 4 and c < self.maze_dims[0] - 1 and not self.maze[r][c]['E']: # Right
            self.player_pos = (r, c + 1)
        # movement == 0 is no-op

    def _calculate_reward_and_termination(self):
        terminated = False
        reward = -0.01  # Small cost for each step

        if self.player_pos == self.exit_pos:
            reward += 10.0 + self.time_remaining # Reward for finishing, bonus for speed
            terminated = True
            self.just_won = True
        elif self.time_remaining <= 0:
            reward -= 5.0 # Penalty for running out of time
            terminated = True
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        maze_w, maze_h = self.maze_dims
        if maze_w == 0 or maze_h == 0: return # Avoid division by zero if maze not generated
        cell_w = self.screen_size[0] / maze_w
        cell_h = self.screen_size[1] / maze_h
        wall_thickness = 2

        # Draw maze walls
        for r in range(maze_h):
            for c in range(maze_w):
                x, y = c * cell_w, r * cell_h
                if self.maze[r][c]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x, y), (x + cell_w, y), wall_thickness)
                if self.maze[r][c]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x, y + cell_h), (x + cell_w, y + cell_h), wall_thickness)
                if self.maze[r][c]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x, y), (x, y + cell_h), wall_thickness)
                if self.maze[r][c]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (x + cell_w, y), (x + cell_w, y + cell_h), wall_thickness)

        # Draw exit
        exit_r, exit_c = self.exit_pos
        exit_x = int(exit_c * cell_w + cell_w / 2)
        exit_y = int(exit_r * cell_h + cell_h / 2)
        exit_radius = int(min(cell_w, cell_h) * 0.35)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_x - exit_radius, exit_y - exit_radius, exit_radius * 2, exit_radius * 2))

        # Draw player
        player_r, player_c = self.player_pos
        player_x = int(player_c * cell_w + cell_w / 2)
        player_y = int(player_r * cell_h + cell_h / 2)
        player_radius = int(min(cell_w, cell_h) * 0.3)
        
        # Glow effect
        glow_radius = int(player_radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_x - glow_radius, player_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Level display
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer display
        timer_text = self.font_ui.render(f"Time: {self.time_remaining:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.screen_size[0] - timer_text.get_width() - 10, 10))
        
        # Game over messages
        if self.game_over:
            if self.just_won:
                msg = "LEVEL COMPLETE!"
                color = self.COLOR_EXIT
            else:
                msg = "TIME UP!"
                color = self.COLOR_PLAYER
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

def validate_implementation(env):
    '''
    Call this to verify implementation.
    '''
    # Test action space
    assert env.action_space.shape == (3,)
    assert env.action_space.nvec.tolist() == [5, 2, 2]
    
    # Test observation space  
    test_obs = env._get_observation()
    assert test_obs.shape == (400, 640, 3)
    assert test_obs.dtype == np.uint8
    
    # Test reset
    obs, info = env.reset()
    assert obs.shape == (400, 640, 3)
    assert isinstance(info, dict)
    
    # Test step
    test_action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(test_action)
    assert obs.shape == (400, 640, 3)
    assert isinstance(reward, (int, float))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    
    print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the video driver for interactive play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    validate_implementation(env)
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        if terminated or truncated:
            # Show final screen for 2 seconds before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses (width, height), but our obs is (height, width, channels)
        # Transpose back for displaying
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Match the environment's internal FPS
        
    env.close()