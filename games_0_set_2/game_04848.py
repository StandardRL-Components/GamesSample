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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. Your goal is to collect all gems and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist maze puzzle. Navigate a procedurally generated maze, collect gems, and find the exit against the clock. The maze gets bigger with each level you clear."
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
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Visuals
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_PATH = (25, 30, 45)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (30, 90, 150)
        self.COLOR_GEM = (255, 220, 50)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_WARN = (255, 80, 80)

        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game progression state
        self.level = 0
        self._initial_maze_dim = 11
        self._initial_time_limit = 100
        self._initial_num_gems = 3

        # Initialize state variables
        self.maze_grid = None
        self.player_pos = None
        self.exit_pos = None
        self.gems = None
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False

        # This will be properly initialized in reset() before use
        self.np_random = None

        # Run validation check
        # self.validate_implementation() # Cannot run here as reset() is needed first
        # Deferring to after first reset in __main__ or by user

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Determine level parameters
        maze_dim = self._initial_maze_dim + self.level * 2
        self.time_limit = self._initial_time_limit + self.level * 20
        self.num_gems = self._initial_num_gems + self.level

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.time_limit

        # Generate maze and place entities
        self.maze_grid, self.start_pos, self.exit_pos = self._generate_maze(maze_dim, maze_dim)
        self.player_pos = list(self.start_pos)
        self._place_gems()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        reward = 0
        old_pos = tuple(self.player_pos)
        old_dist_to_exit = self._manhattan_distance(old_pos, self.exit_pos)

        # Update player position based on movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            # Check for wall collision
            if self.maze_grid[next_pos[1]][next_pos[0]] == 0:
                self.player_pos = list(next_pos)

        # Update game logic
        self.steps += 1
        self.time_remaining -= 1

        # Check for gem collection
        if tuple(self.player_pos) in self.gems:
            self.gems.remove(tuple(self.player_pos))
            reward += 5.0
            # sfx: gem collect sound

        # Calculate continuous reward
        new_dist_to_exit = self._manhattan_distance(tuple(self.player_pos), self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > old_dist_to_exit:
            reward -= 0.01

        self.score += reward
        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if tuple(self.player_pos) == self.exit_pos:
                reward += 50.0
                self.score += 50.0
                self.level += 1  # Progress to next level on next reset
                # sfx: level complete fanfare
            elif self.time_remaining <= 0:
                reward -= 10.0
                self.score -= 10.0
                # sfx: game over sad sound

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _check_termination(self):
        if tuple(self.player_pos) == self.exit_pos and not self.gems:
            self.game_over = True
            return True
        if self.time_remaining <= 0:
            self.game_over = True
            return True
        if self.steps >= 1000: # Timeout to prevent infinite episodes
            self.game_over = True
            return True
        return False

    def _generate_maze(self, width, height):
        # Ensure odd dimensions for perfect maze generation
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1

        # 0 = path, 1 = wall
        grid = np.ones((height, width), dtype=np.uint8)

        # Recursive backtracking
        stack = []
        start_x, start_y = (1, 1)
        grid[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # Select a random neighbor
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]

                # Carve path
                grid[ny, nx] = 0
                grid[y + (ny - y) // 2, x + (nx - x) // 2] = 0

                stack.append((nx, ny))
            else:
                stack.pop()

        start_pos = (1, 1)
        exit_pos = (width - 2, height - 2)

        return grid, start_pos, exit_pos

    def _place_gems(self):
        self.gems = set()
        path_cells = np.argwhere(self.maze_grid == 0).tolist()

        # Remove start and end positions from potential gem locations
        if list(self.start_pos) in path_cells:
            path_cells.remove(list(self.start_pos))
        if list(self.exit_pos) in path_cells:
            path_cells.remove(list(self.exit_pos))

        if len(path_cells) < self.num_gems:
            # Fallback if not enough space, though unlikely
            num_to_place = len(path_cells)
        else:
            num_to_place = self.num_gems

        if num_to_place > 0:
            indices = self.np_random.choice(len(path_cells), num_to_place, replace=False)
            for i in indices:
                # path_cells are [y, x], convert to (x, y) tuple for consistency
                self.gems.add((path_cells[i][1], path_cells[i][0]))

    def _get_observation(self):
        # Clear screen with path color first, then draw walls
        self.screen.fill(self.COLOR_PATH)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if self.maze_grid is None:
            return
            
        maze_h, maze_w = self.maze_grid.shape

        # Calculate cell size and centering offset
        cell_w = self.width / maze_w
        cell_h = self.height / maze_h

        # Draw walls
        for y in range(maze_h):
            for x in range(maze_w):
                if self.maze_grid[y][x] == 1:
                    pygame.draw.rect(
                        self.screen, self.COLOR_WALL,
                        (x * cell_w, y * cell_h, math.ceil(cell_w), math.ceil(cell_h))
                    )

        # Draw exit
        exit_x, exit_y = self.exit_pos
        pygame.draw.rect(
            self.screen, self.COLOR_EXIT,
            (exit_x * cell_w, exit_y * cell_h, cell_w, cell_h)
        )

        # Draw gems
        for gem_x, gem_y in self.gems:
            gem_center_x = int((gem_x + 0.5) * cell_w)
            gem_center_y = int((gem_y + 0.5) * cell_h)
            gem_size = int(min(cell_w, cell_h) * 0.3)
            pygame.draw.rect(
                self.screen, self.COLOR_GEM,
                (gem_center_x - gem_size, gem_center_y - gem_size, gem_size * 2, gem_size * 2)
            )

        # Draw player
        player_center_x = int((self.player_pos[0] + 0.5) * cell_w)
        player_center_y = int((self.player_pos[1] + 0.5) * cell_h)
        player_radius = int(min(cell_w, cell_h) * 0.35)

        # Use gfxdraw for smooth, anti-aliased circle
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time remaining
        time_color = self.COLOR_TEXT if self.time_remaining > self.time_limit * 0.2 else self.COLOR_TEXT_WARN
        time_text = self.font_main.render(f"TIME: {self.time_remaining}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        self.screen.blit(score_text, score_rect)

        # Gems remaining
        gem_text = self.font_main.render(f"GEMS: {len(self.gems)}", True, self.COLOR_GEM)
        gem_rect = gem_text.get_rect(midtop=(self.width // 2, 10))
        self.screen.blit(gem_text, gem_rect)

        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level + 1}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(bottomleft=(10, self.height - 10))
        self.screen.blit(level_text, level_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining,
            "gems_remaining": len(self.gems),
        }

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
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Create a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.width, env.height))

    total_reward = 0
    total_steps = 0

    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Reset the game if 'r' is pressed after game over
                if env.game_over and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    total_steps = 0
                    print("\n--- New Game ---")

        if not env.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            # Since auto_advance is False, we only step when a key is pressed
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1

                print(
                    f"Step: {total_steps}, Action: {action[0]}, Reward: {reward:.2f}, "
                    f"Total Reward: {total_reward:.2f}, Done: {terminated}"
                )

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate of the human-playable version
        env.clock.tick(30)

    env.close()