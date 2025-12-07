import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit before time or moves run out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist maze-solving game. Find the exit in a procedurally generated "
        "labyrinth under time and move constraints. The maze grows larger as you succeed."
    )

    # Should frames auto-advance or wait for user input?
    # auto_advance=True is chosen due to the real-time limit.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.W, self.H = 640, 400
        self.FPS = 30  # Used for time calculations

        # Game constants
        self.MAX_TIME = 60
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1800  # 60 seconds * 30 fps

        # Visuals
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_EXIT_GLOW = (200, 255, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables (persistent across episodes)
        self.successful_episodes = 0
        self.maze_dim_start = 5

        # State variables (reset each episode)
        self.maze_dim = self.maze_dim_start
        self.maze = []
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.time_left = 0
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.move_cooldown = 0
        self.MOVE_COOLDOWN_FRAMES = 5  # 1/6th of a second cooldown

        self.reset()
        # self.validate_implementation() # Optional validation call

    def _generate_maze(self, width, height):
        # Maze represented as a grid of cells. Each cell is a dict of its walls.
        # {'N': 1, 'S': 1, 'E': 1, 'W': 1} where 1 is a wall, 0 is a path.
        # Maze indexing is maze[y][x]
        maze = [[{'N': 1, 'S': 1, 'E': 1, 'W': 1} for _ in range(width)] for _ in range(height)]

        # Use recursive backtracking (DFS) to carve a perfect maze
        stack = []
        visited = set()

        # Start at top-left
        start_cell = (0, 0)
        stack.append(start_cell)
        visited.add(start_cell)

        while stack:
            cx, cy = stack[-1]

            # Find unvisited neighbors
            neighbors = []
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append(('N', cx, cy - 1))
            if cy < height - 1 and (cx, cy + 1) not in visited: neighbors.append(('S', cx, cy + 1))
            if cx < width - 1 and (cx + 1, cy) not in visited: neighbors.append(('E', cx + 1, cy))
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append(('W', cx - 1, cy))

            if neighbors:
                # Choose a random neighbor using an index, to avoid type casting issues with np.random.choice
                idx = self.np_random.integers(len(neighbors))
                direction, nx, ny = neighbors[idx]

                # Carve path to neighbor
                if direction == 'N':
                    maze[cy][cx]['N'] = 0
                    maze[ny][nx]['S'] = 0
                elif direction == 'S':
                    maze[cy][cx]['S'] = 0
                    maze[ny][nx]['N'] = 0
                elif direction == 'E':
                    maze[cy][cx]['E'] = 0
                    maze[ny][nx]['W'] = 0  # FIX: Correct indexing from [nx][ny] to [ny][nx]
                elif direction == 'W':
                    maze[cy][cx]['W'] = 0
                    maze[ny][nx]['E'] = 0  # FIX: Correct indexing from [nx][ny] to [ny][nx]

                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Increase difficulty every 5 successful runs
        level = self.successful_episodes // 5
        self.maze_dim = self.maze_dim_start + level

        # Generate maze
        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.move_cooldown = 0

        self.player_pos = [0, 0]
        self.exit_pos = [self.maze_dim - 1, self.maze_dim - 1]

        self.time_left = self.MAX_TIME
        self.moves_left = self.MAX_MOVES

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no more actions, just advance frame
            # The environment is terminated, so the agent should reset it.
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        # Update game logic
        self.steps += 1
        self.time_left -= 1.0 / self.FPS
        reward = -0.1  # Small penalty for each step (time passing)

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # Process movement if not on cooldown and a move is requested
        if movement != 0 and self.move_cooldown == 0:
            px, py = self.player_pos
            next_pos = list(self.player_pos)

            # Action mapping: 1=up, 2=down, 3=left, 4=right
            if movement == 1 and py > 0 and self.maze[py][px]['N'] == 0:
                next_pos[1] -= 1
            elif movement == 2 and py < self.maze_dim - 1 and self.maze[py][px]['S'] == 0:
                next_pos[1] += 1
            elif movement == 3 and px > 0 and self.maze[py][px]['W'] == 0:
                next_pos[0] -= 1
            elif movement == 4 and px < self.maze_dim - 1 and self.maze[py][px]['E'] == 0:
                next_pos[0] += 1

            # If move was valid, update state
            if next_pos != self.player_pos:
                self.player_pos = next_pos
                self.moves_left -= 1
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
                # No immediate reward for moving, just the resource cost

        # Check for termination conditions
        terminated = False
        player_at_exit = self.player_pos == self.exit_pos

        if player_at_exit:
            terminated = True
            self.game_over = True
            reward += 5.0  # Base reward for reaching exit
            if self.moves_left >= 0 and self.time_left > 0:
                reward += 10.0  # Bonus for winning within limits
                self.successful_episodes += 1
                self.win_message = "YOU WIN!"
            else:
                self.win_message = "EXIT REACHED"
        elif self.moves_left < 0:
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF MOVES"
        elif self.time_left <= 0:
            terminated = True
            self.game_over = True
            self.win_message = "TIME'S UP"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "STEP LIMIT"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _render_game(self):
        # Calculate rendering parameters to fit maze on screen
        padding = 60
        maze_render_w = self.W - 2 * padding
        maze_render_h = self.H - 2 * padding

        cell_size = min(maze_render_w / self.maze_dim, maze_render_h / self.maze_dim)

        offset_x = (self.W - self.maze_dim * cell_size) / 2
        offset_y = (self.H - self.maze_dim * cell_size) / 2

        wall_thickness = max(1, int(cell_size * 0.1))

        # Render maze walls
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                cell = self.maze[y][x]
                px, py = offset_x + x * cell_size, offset_y + y * cell_size

                if cell['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + cell_size, py), wall_thickness)
                if cell['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + cell_size), wall_thickness)

        # Draw outer boundaries
        bound_x = offset_x + self.maze_dim * cell_size
        bound_y = offset_y + self.maze_dim * cell_size
        pygame.draw.line(self.screen, self.COLOR_WALL, (bound_x, offset_y), (bound_x, bound_y), wall_thickness)
        pygame.draw.line(self.screen, self.COLOR_WALL, (offset_x, bound_y), (bound_x, bound_y), wall_thickness)

        # Render Exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(
            offset_x + ex * cell_size, offset_y + ey * cell_size, cell_size, cell_size
        )
        glow_radius = int(cell_size)
        pygame.draw.circle(self.screen, self.COLOR_EXIT_GLOW, exit_rect.center, glow_radius, width=int(cell_size / 2.5))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-cell_size * 0.3, -cell_size * 0.3))

        # Render Player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            offset_x + px * cell_size, offset_y + py * cell_size, cell_size, cell_size
        )
        glow_radius = int(cell_size * 0.8)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_GLOW, player_rect.center, glow_radius,
                           width=int(cell_size / 3))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-cell_size * 0.4, -cell_size * 0.4))

    def _render_ui(self):
        # UI background
        ui_bar = pygame.Surface((self.W, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # Moves Left
        moves_text = self.font_small.render(f"MOVES: {max(0, self.moves_left)}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        # Time Left
        time_str = f"TIME: {max(0, self.time_left):.1f}"
        time_text = self.font_small.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.W - 15, 10))
        self.screen.blit(time_text, time_rect)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.W / 2, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, end_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        # Pygame surface is (W, H), but observation space is (H, W)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "moves_left": self.moves_left,
            "successful_episodes": self.successful_episodes,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")