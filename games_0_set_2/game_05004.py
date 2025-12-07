import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import collections
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the cavern. "
        "Reach the green exit tile before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated cavern, solving tile-based puzzles to escape before your moves run out. "
        "Blue tiles grant moves, red tiles are traps, and purple tiles teleport you."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Tile Types
    TILE_WALL = 0
    TILE_FLOOR = 1
    TILE_BONUS = 2
    TILE_TRAP = 3
    TILE_TELEPORT = 4
    TILE_EXIT = 5

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 35, 55)
    COLOR_FLOOR = (70, 65, 90)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_EXIT = (0, 255, 128)
    COLOR_BONUS = (0, 128, 255)
    COLOR_TRAP = (255, 50, 50)
    COLOR_TELEPORT = (170, 0, 255)
    COLOR_UI_TEXT = (240, 240, 240)

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)

        self.level = 1
        self.grid = np.array([[]])
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.moves_left = 0
        self.initial_moves = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animations = []

        # self.reset() is called here in the original code, but it's better to
        # let the user/runner call it explicitly for the first time.
        # However, to match the failing behavior and fix it, we'll keep it.
        # In a standard Gym setup, the first call is `env.reset()`.
        # The validation at the end will call reset anyway.
        # self.reset() # This was causing the crash during initialization.
        self.validate_implementation_requires_reset = True


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animations = []

        if options and "level" in options:
            self.level = options.get("level", 1)

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        prev_dist = self._manhattan_distance(self.player_pos, self.exit_pos)

        # Movement action
        if movement > 0:
            self.moves_left -= 1
            px, py = self.player_pos
            nx, ny = px, py

            if movement == 1: ny -= 1  # Up
            elif movement == 2: ny += 1  # Down
            elif movement == 3: nx -= 1  # Left
            elif movement == 4: nx += 1  # Right

            if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1] and self.grid[ny][nx] != self.TILE_WALL:
                self.player_pos = (nx, ny)

                # Reward for moving closer/further
                new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
                reward += (prev_dist - new_dist) * 0.1

                # Handle tile effects
                tile_type = self.grid[ny][nx]
                if tile_type != self.TILE_FLOOR:
                    self.grid[ny][nx] = self.TILE_FLOOR  # Consume the tile

                    if tile_type == self.TILE_EXIT:
                        reward += 100
                        self.game_over = True
                        self._add_animation(self.player_pos, self.COLOR_EXIT, 20, 30)
                        self.level += 1  # Progress difficulty for next reset
                    elif tile_type == self.TILE_BONUS:
                        reward += 5
                        self.moves_left = min(self.initial_moves, self.moves_left + 5)
                        self._add_animation(self.player_pos, self.COLOR_BONUS, 15, 20)
                    elif tile_type == self.TILE_TRAP:
                        reward -= 10
                        self.moves_left -= 10
                        self._add_animation(self.player_pos, self.COLOR_TRAP, 15, 20)
                    elif tile_type == self.TILE_TELEPORT:
                        self._add_animation(self.player_pos, self.COLOR_TELEPORT, 15, 20)
                        floor_tiles = [(x, y) for y, row in enumerate(self.grid) for x, tile in enumerate(row) if tile == self.TILE_FLOOR]
                        if floor_tiles:
                            # np.random.choice on a list of tuples requires an extra argument
                            idx = self.np_random.integers(len(floor_tiles))
                            self.player_pos = floor_tiles[idx]
                        self._add_animation(self.player_pos, self.COLOR_TELEPORT, 15, 20)
            else:
                # Penalty for hitting a wall
                reward -= 0.5

        # Check for termination
        terminated = self.game_over
        if not terminated:
            if self.moves_left <= 0:
                reward -= 50
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.game_over = terminated
        self.score += reward
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        # Scale difficulty with level
        grid_w = min(41, 15 + (self.level - 1) * 2)
        grid_h = min(25, 9 + (self.level - 1) * 2)

        self.grid = np.full((grid_h, grid_w), self.TILE_WALL, dtype=np.uint8)

        # Maze generation using recursive backtracking (DFS)
        stack = []
        # FIX: Generate start_x and start_y with correct bounds
        start_x = self.np_random.integers(1, grid_w - 1)
        start_y = self.np_random.integers(1, grid_h - 1)
        
        if start_x % 2 == 0: start_x -= 1
        if start_y % 2 == 0: start_y -= 1
        
        # Ensure start is within bounds after adjustment
        start_x = max(1, start_x)
        start_y = max(1, start_y)

        self.grid[start_y][start_x] = self.TILE_FLOOR
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < grid_w - 1 and 0 < ny < grid_h - 1 and self.grid[ny][nx] == self.TILE_WALL:
                    neighbors.append((nx, ny))

            if neighbors:
                # Correctly sample from a list of tuples
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                self.grid[ny][nx] = self.TILE_FLOOR
                self.grid[y + (ny - y) // 2][x + (nx - x) // 2] = self.TILE_FLOOR
                stack.append((nx, ny))
            else:
                stack.pop()

        # Find all floor tiles to place items
        floor_tiles = []
        for y in range(grid_h):
            for x in range(grid_w):
                if self.grid[y][x] == self.TILE_FLOOR:
                    floor_tiles.append((x, y))

        self.np_random.shuffle(floor_tiles)

        # Place player and exit far apart
        self.player_pos = floor_tiles.pop(0)

        # Use BFS to find the furthest point for the exit
        queue = collections.deque([(self.player_pos, 0)])
        visited = {self.player_pos}
        farthest_node, max_dist = self.player_pos, 0

        while queue:
            (vx, vy), dist = queue.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_node = (vx, vy)

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = vx + dx, vy + dy
                # Check grid bounds and tile type for valid neighbors
                if 0 <= ny < grid_h and 0 <= nx < grid_w and self.grid[ny][nx] != self.TILE_WALL and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        self.exit_pos = farthest_node
        if self.exit_pos in floor_tiles:
            floor_tiles.remove(self.exit_pos)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = self.TILE_EXIT

        # Set initial moves based on optimal path length + buffer
        self.initial_moves = int(max_dist * 1.8) + 10
        self.moves_left = self.initial_moves

        # Place special tiles
        num_bonus = max(1, 5 - (self.level - 1) // 2)
        num_traps = 2 + (self.level - 1) // 2
        num_teleports = 1 + (self.level - 1) // 3

        for _ in range(num_bonus):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            self.grid[pos[1]][pos[0]] = self.TILE_BONUS
        for _ in range(num_traps):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            self.grid[pos[1]][pos[0]] = self.TILE_TRAP
        for _ in range(num_teleports):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            self.grid[pos[1]][pos[0]] = self.TILE_TELEPORT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if self.grid.size == 0: return # Don't render if grid not initialized
        grid_h, grid_w = self.grid.shape

        # Calculate tile size and centering offset
        tile_size = min(
            (self.SCREEN_WIDTH - 40) // grid_w,
            (self.SCREEN_HEIGHT - 60) // grid_h
        )
        offset_x = (self.SCREEN_WIDTH - grid_w * tile_size) // 2
        offset_y = (self.SCREEN_HEIGHT - grid_h * tile_size) // 2

        # Draw grid
        for y, row in enumerate(self.grid):
            for x, tile_type in enumerate(row):
                rect = pygame.Rect(offset_x + x * tile_size, offset_y + y * tile_size, tile_size, tile_size)

                color = self.COLOR_FLOOR
                if tile_type == self.TILE_WALL: color = self.COLOR_WALL
                elif tile_type == self.TILE_EXIT: color = self.COLOR_EXIT
                elif tile_type == self.TILE_BONUS: color = self.COLOR_BONUS
                elif tile_type == self.TILE_TRAP: color = self.COLOR_TRAP
                elif tile_type == self.TILE_TELEPORT: color = self.COLOR_TELEPORT
                
                pygame.draw.rect(self.screen, color, rect)
                
                if tile_type == self.TILE_EXIT:
                    # Glowing Exit
                    t = self.steps % 30 / 30.0
                    glow_size = int(tile_size * (1.5 + 0.2 * math.sin(t * 2 * math.pi)))
                    glow_alpha = 100 + 50 * math.sin(t * 2 * math.pi)
                    glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*self.COLOR_EXIT, glow_alpha), (glow_size // 2, glow_size // 2), glow_size // 2)
                    self.screen.blit(glow_surf, glow_surf.get_rect(center=rect.center))


        # Draw animations
        for anim in self.animations[:]:
            anim['timer'] -= 1
            if anim['timer'] <= 0:
                self.animations.remove(anim)
                continue

            x, y = anim['pos']
            center_x = offset_x + int((x + 0.5) * tile_size)
            center_y = offset_y + int((y + 0.5) * tile_size)

            radius = int(tile_size * (anim['max_radius'] / anim['duration']) * (anim['timer'] / anim['duration']))
            alpha = int(255 * (anim['timer'] / anim['duration']))

            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*anim['color'], alpha), (radius, radius), radius)
            self.screen.blit(s, s.get_rect(center=(center_x, center_y)))

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(offset_x + px * tile_size, offset_y + py * tile_size, tile_size, tile_size)

        # Player "breathing" effect
        t = self.steps % 60 / 60.0
        scale = 0.8 + 0.1 * math.sin(t * 2 * math.pi)
        scaled_size = int(tile_size * scale)
        scaled_rect = pygame.Rect(0, 0, scaled_size, scaled_size)
        scaled_rect.center = player_rect.center
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, scaled_rect, border_radius=3)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_left}"
        level_text = f"Cavern: {self.level}"

        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        level_surf = self.font_ui.render(level_text, True, self.COLOR_UI_TEXT)

        self.screen.blit(moves_surf, (20, 10))
        self.screen.blit(level_surf, (self.SCREEN_WIDTH - level_surf.get_width() - 20, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _add_animation(self, pos, color, max_radius, duration):
        self.animations.append({
            'pos': pos, 'color': color, 'max_radius': max_radius,
            'duration': duration, 'timer': duration
        })

    def validate_implementation(self):
        # This validation is for internal use and can be removed.
        # It was part of the original code to help diagnose issues.
        if getattr(self, "validate_implementation_requires_reset", False):
            print("Calling reset() before validation...")
            self.reset()
            self.validate_implementation_requires_reset = False

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
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Pygame display must be initialized in the main thread
    pygame.display.init()
    pygame.font.init()

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Cavern Explorer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r:  # Press R to reset
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

                if action[0] != 0:
                    if not terminated:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}, Terminated: {terminated}")
                    else:
                        print("Game over. Press 'R' to reset.")


        # Draw the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(30)

    pygame.quit()