import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move your monster through the maze. "
        "Reach the green exit before you run out of moves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A puzzle game where you guide a monster through a procedurally generated maze. "
        "Each move costs, so find the most efficient path to the exit. "
        "The mazes get larger and more complex as you succeed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (240, 240, 255)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_EXIT = (50, 255, 100)
        self.COLOR_TRAIL = (120, 20, 20)
        self.COLOR_UI = (220, 220, 220)

        # Game progression state
        self.initial_maze_dim = 5
        self.mazes_per_stage = 3
        self.stage = 0
        self.completed_in_stage = 0

        # Game state variables (initialized in reset)
        self.maze = None
        self.maze_width = 0
        self.maze_height = 0
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.moves_remaining = 0
        self.path_trail = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state
        # The reset call is deferred to the first call by the user/runner
        # as per the standard Gymnasium API. However, some internal state
        # needs to be initialized for methods like _get_observation to work
        # before the first reset.
        self.maze_width = self.initial_maze_dim
        self.maze_height = self.initial_maze_dim
        self._generate_maze(self.maze_width, self.maze_height)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Handle progression
        if self.completed_in_stage >= self.mazes_per_stage:
            self.stage += 1
            self.completed_in_stage = 0

        self.maze_width = self.initial_maze_dim + self.stage * 2
        self.maze_height = self.initial_maze_dim + self.stage * 2

        # Generate a new maze and determine start/end points
        self._generate_maze(self.maze_width, self.maze_height)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.path_trail = [self.player_pos]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        terminated = False
        
        moved = False
        if movement != 0:  # 0 is no-op
            moved = self._move_player(movement)

        if moved:
            self.moves_remaining -= 1
            reward = -0.1
            self.path_trail.append(self.player_pos)
            if len(self.path_trail) > 100:  # Limit trail length for performance
                self.path_trail.pop(0)
            # Sound: Step sfx

        # Check win condition
        if self.player_pos == self.exit_pos:
            reward = 10.0
            self.score = reward
            terminated = True
            self.completed_in_stage += 1
            # Sound: Victory fanfare!

        # Check lose condition
        elif self.moves_remaining <= 0:
            reward = -10.0
            self.score = reward
            terminated = True
            # Sound: Failure buzz

        self.steps += 1
        if self.steps >= 1000:
            # Adding a truncated condition for long episodes, although terminated is used
            terminated = True 

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this logic
            self._get_info()
        )

    def _move_player(self, movement):
        r, c = self.player_pos
        # Wall bitmasks: N=1, S=2, E=4, W=8
        if movement == 1 and not (self.maze[r][c] & 1): # Up
            self.player_pos = (r - 1, c)
            return True
        elif movement == 2 and not (self.maze[r][c] & 2): # Down
            self.player_pos = (r + 1, c)
            return True
        elif movement == 3 and not (self.maze[r][c] & 8): # Left
            self.player_pos = (r, c - 1)
            return True
        elif movement == 4 and not (self.maze[r][c] & 4): # Right
            self.player_pos = (r, c + 1)
            return True
        return False

    def _generate_maze(self, width, height):
        # Wall bitmasks: N=1, S=2, E=4, W=8
        self.maze = np.full((height, width), 15, dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        stack = [(self.np_random.integers(0, height), self.np_random.integers(0, width))]
        visited[stack[0]] = True

        while stack:
            r, c = stack[-1]
            neighbors = []
            # Check North
            if r > 0 and not visited[r - 1, c]: neighbors.append((r - 1, c, 1, 2))
            # Check South
            if r < height - 1 and not visited[r + 1, c]: neighbors.append((r + 1, c, 2, 1))
            # Check East
            if c < width - 1 and not visited[r, c + 1]: neighbors.append((r, c + 1, 4, 8))
            # Check West
            if c > 0 and not visited[r, c - 1]: neighbors.append((r, c - 1, 8, 4))

            if neighbors:
                idx = self.np_random.integers(0, len(neighbors))
                nr, nc, wall, opposite_wall = neighbors[idx]
                
                # FIX: Use subtraction instead of bitwise NOT to avoid overflow with uint8.
                # This correctly removes the wall flag.
                self.maze[r, c] -= wall
                self.maze[nr, nc] -= opposite_wall
                
                visited[nr, nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        # Find the longest path to set start and exit points
        dist_from_corner, _ = self._bfs((0, 0))
        farthest_node, _ = max(dist_from_corner.items(), key=lambda item: item[1])
        
        distances, path_lengths = self._bfs(farthest_node)
        # In case of disconnected maze parts, handle key error
        if not distances:
             # Fallback if BFS fails (e.g., 1x1 maze)
             start_node, end_node = (0,0), (0,0)
        else:
             start_node, _ = max(distances.items(), key=lambda item: item[1])
             end_node = farthest_node
        
        self.player_pos = start_node
        self.exit_pos = end_node
        
        path_len = path_lengths.get(end_node, 1)
        self.moves_remaining = math.ceil(path_len * 1.5) + 5

    def _bfs(self, start_node):
        q = deque([(start_node, 0)])
        visited = {start_node}
        distances = {start_node: 0}
        path_lengths = {start_node: 0}

        while q:
            (r, c), dist = q.popleft()
            
            # Check neighbors
            # N
            if r > 0 and not (self.maze[r, c] & 1) and (r - 1, c) not in visited:
                n = (r - 1, c)
                visited.add(n)
                distances[n] = dist + 1
                path_lengths[n] = path_lengths[(r,c)] + 1
                q.append((n, dist + 1))
            # S
            if r < self.maze_height - 1 and not (self.maze[r, c] & 2) and (r + 1, c) not in visited:
                n = (r + 1, c)
                visited.add(n)
                distances[n] = dist + 1
                path_lengths[n] = path_lengths[(r,c)] + 1
                q.append((n, dist + 1))
            # E
            if c < self.maze_width - 1 and not (self.maze[r, c] & 4) and (r, c + 1) not in visited:
                n = (r, c + 1)
                visited.add(n)
                distances[n] = dist + 1
                path_lengths[n] = path_lengths[(r,c)] + 1
                q.append((n, dist + 1))
            # W
            if c > 0 and not (self.maze[r, c] & 8) and (r, c - 1) not in visited:
                n = (r, c - 1)
                visited.add(n)
                distances[n] = dist + 1
                path_lengths[n] = path_lengths[(r,c)] + 1
                q.append((n, dist + 1))
        
        return distances, path_lengths

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        padding = 40
        drawable_width = self.screen_width - 2 * padding
        drawable_height = self.screen_height - 2 * padding
        
        cell_w = drawable_width / self.maze_width
        cell_h = drawable_height / self.maze_height

        offset_x = padding
        offset_y = padding

        # Render Trail
        if len(self.path_trail) > 1:
            for i, pos in enumerate(self.path_trail):
                r, c = pos
                center_x = int(offset_x + (c + 0.5) * cell_w)
                center_y = int(offset_y + (r + 0.5) * cell_h)
                radius = int(min(cell_w, cell_h) * 0.15)
                
                # Fade trail based on age
                trail_color = (
                    max(0, self.COLOR_TRAIL[0] - (len(self.path_trail) - i) * 2),
                    self.COLOR_TRAIL[1],
                    self.COLOR_TRAIL[2]
                )
                if radius > 0:
                     pygame.draw.circle(self.screen, trail_color, (center_x, center_y), radius)

        # Render Walls
        for r in range(self.maze_height):
            for c in range(self.maze_width):
                x1, y1 = int(offset_x + c * cell_w), int(offset_y + r * cell_h)
                x2, y2 = int(offset_x + (c + 1) * cell_w), int(offset_y + (r + 1) * cell_h)
                
                if self.maze[r, c] & 1: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x2, y1), 2) # North
                if self.maze[r, c] & 2: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y2), (x2, y2), 2) # South
                if self.maze[r, c] & 4: pygame.draw.line(self.screen, self.COLOR_WALL, (x2, y1), (x2, y2), 2) # East
                if self.maze[r, c] & 8: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x1, y2), 2) # West

        # Render Exit
        exit_r, exit_c = self.exit_pos
        exit_rect = pygame.Rect(
            offset_x + exit_c * cell_w + cell_w * 0.1,
            offset_y + exit_r * cell_h + cell_h * 0.1,
            cell_w * 0.8,
            cell_h * 0.8
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=int(min(cell_w, cell_h)*0.2))

        # Render Player
        player_r, player_c = self.player_pos
        player_center_x = int(offset_x + (player_c + 0.5) * cell_w)
        player_center_y = int(offset_y + (player_r + 0.5) * cell_h)
        player_radius = int(min(cell_w, cell_h) * 0.35)

        if player_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}"
        text_surface = self.font.render(moves_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (15, 10))

        stage_text = f"Maze: {self.stage * self.mazes_per_stage + self.completed_in_stage + 1}"
        text_surface = self.font.render(stage_text, True, self.COLOR_UI)
        text_rect = text_surface.get_rect(topright=(self.screen_width - 15, 10))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "stage": self.stage,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # This part is for human testing and requires a display.
    # To run, you might need to comment out the os.environ line at the top.
    
    # --- Manual Play ---
    try:
        # Check if we are in a headless environment
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            print("Running in headless mode. Skipping manual play test.")
            # Simple API test
            env = GameEnv()
            obs, info = env.reset()
            assert isinstance(obs, np.ndarray)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            print("Headless API test passed.")
            env.close()
        else:
            raise KeyError # Fallback to interactive mode
    except (KeyError, pygame.error):
        # Interactive mode
        print("Running manual play test...")
        # Re-initialize pygame with default video driver
        pygame.quit()
        pygame.init()
        
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Maze Runner")
        
        obs, info = env.reset()
        terminated = False
        
        running = True
        while running:
            movement = 0 # No-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_r: # Reset on 'r' key
                        obs, info = env.reset()
                        terminated = False
                    
                    if terminated and event.key != pygame.K_r:
                        continue # Don't step if game is over, unless resetting

                    action = [movement, 0, 0]
                    
                    if movement != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Moves Left: {info['moves_remaining']}, Terminated: {terminated}")
            
            # Draw the observation to the display screen
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(draw_surface, (0, 0))
            pygame.display.flip()
            
            if terminated:
                # Display "Game Over" message
                font = pygame.font.SysFont("monospace", 48, bold=True)
                text = font.render("GAME OVER", True, (255, 0, 0))
                text_rect = text.get_rect(center=(env.screen_width/2, env.screen_height/2 - 20))
                screen.blit(text, text_rect)
                
                font_small = pygame.font.SysFont("monospace", 24, bold=True)
                text_small = font_small.render("Press 'r' to restart", True, (220, 220, 220))
                text_small_rect = text_small.get_rect(center=(env.screen_width/2, env.screen_height/2 + 20))
                screen.blit(text_small, text_small_rect)
                
                pygame.display.flip()

            pygame.time.wait(30)
            
        env.close()