
# Generated: 2025-08-27T21:49:28.562502
# Source Brief: brief_02920.md
# Brief Index: 2920

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Your goal is to collect all the gold keys and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down maze puzzle. Find all the keys and escape before the timer hits zero. "
        "The maze gets larger and more complex with each successful run."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_MAZE_DIMS = 25
        self.MAX_TIME = 300
        self.MAX_KEYS = 5
        self.MAX_EPISODE_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 80, 100)
        self.COLOR_PATH = (40, 60, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_KEY = (255, 215, 0)
        self.COLOR_KEY_GLOW = (255, 235, 100)
        self.COLOR_EXIT_LOCKED = (180, 40, 40)
        self.COLOR_EXIT_UNLOCKED = (40, 180, 40)
        self.COLOR_UI_BG = (0, 0, 0, 180)
        self.COLOR_UI_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State Initialization ---
        self.maze_dims = np.array([5, 5])
        self.time_limit = 50
        self.num_keys = 1
        self.last_game_won = False

        self.maze = None
        self.player_pos = None
        self.key_positions = None
        self.exit_pos = None
        self.keys_collected_count = 0
        self.exit_unlocked = False
        self.steps = 0
        self.steps_remaining = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.last_game_won:
            self.maze_dims = np.minimum(self.maze_dims + 1, self.MAX_MAZE_DIMS)
            self.time_limit = min(self.time_limit + 10, self.MAX_TIME)
            if self.maze_dims[0] % 4 == 0 and self.num_keys < self.MAX_KEYS:
                 self.num_keys += 1
        self.last_game_won = False

        self.maze = self._generate_maze(self.maze_dims[0], self.maze_dims[1])
        self._place_objects()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.keys_collected_count = 0
        self.exit_unlocked = False
        self.steps_remaining = self.time_limit

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty for taking a step
        self.steps += 1
        self.steps_remaining -= 1

        # --- Player Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            new_pos = self.player_pos + np.array([dx, dy])
            if self.maze[new_pos[1], new_pos[0]] == 0:  # 0 is path
                self.player_pos = new_pos
                # sound_placeholder: player_move.wav

        # --- Check for Key Collection ---
        collected_key_index = -1
        for i, key_pos in enumerate(self.key_positions):
            if np.array_equal(self.player_pos, key_pos):
                collected_key_index = i
                break
        
        if collected_key_index != -1:
            self.key_positions.pop(collected_key_index)
            self.keys_collected_count += 1
            reward += 10
            self.score += 10
            # sound_placeholder: key_collect.wav
            if self.keys_collected_count == self.num_keys:
                self.exit_unlocked = True
                # sound_placeholder: exit_unlocked.wav

        # --- Check Termination Conditions ---
        terminated = False
        if self.exit_unlocked and np.array_equal(self.player_pos, self.exit_pos):
            reward += 100
            self.score += 100
            terminated = True
            self.last_game_won = True
            # sound_placeholder: win_level.wav
        elif self.steps_remaining <= 0:
            reward -= 10
            self.score -= 10
            terminated = True
            # sound_placeholder: lose_timeout.wav
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True

        self.game_over = terminated
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_maze(self, width, height):
        # Maze grid: 1 for wall, 0 for path.
        # Grid size is (2*h+1) x (2*w+1) to represent walls between cells.
        shape = (2 * height + 1, 2 * width + 1)
        maze = np.ones(shape, dtype=np.int8)
        
        # Carving function using recursive backtracking
        def carve(x, y):
            maze[y, x] = 0
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.np_random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + 2 * dx, y + 2 * dy
                if 0 < nx < shape[1] and 0 < ny < shape[0] and maze[ny, nx] == 1:
                    maze[y + dy, x + dx] = 0
                    carve(nx, ny)

        # Start carving from a random odd-indexed cell
        start_x = self.np_random.integers(0, width) * 2 + 1
        start_y = self.np_random.integers(0, height) * 2 + 1
        carve(start_x, start_y)
        return maze

    def _place_objects(self):
        # Get all valid path coordinates (odd-indexed)
        path_coords = []
        for r in range(self.maze_dims[1]):
            for c in range(self.maze_dims[0]):
                path_coords.append(np.array([2 * c + 1, 2 * r + 1]))

        while True:
            # Randomly select distinct positions for all objects
            indices = self.np_random.choice(len(path_coords), size=self.num_keys + 2, replace=False)
            selected_coords = [path_coords[i] for i in indices]

            self.player_pos = selected_coords.pop()
            self.exit_pos = selected_coords.pop()
            self.key_positions = selected_coords

            # --- Solvability Check (BFS) ---
            q = [self.player_pos]
            visited = {tuple(self.player_pos)}
            
            head = 0
            while head < len(q):
                curr = q[head]
                head += 1
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = curr + np.array([dx, dy])
                    if self.maze[neighbor[1], neighbor[0]] == 0 and tuple(neighbor) not in visited:
                        visited.add(tuple(neighbor))
                        q.append(neighbor)
            
            all_reachable = True
            for pos in self.key_positions + [self.exit_pos]:
                if tuple(pos) not in visited:
                    all_reachable = False
                    break
            
            if all_reachable:
                break # Found a solvable layout

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
            "steps_remaining": self.steps_remaining,
            "keys_collected": self.keys_collected_count,
            "total_keys": self.num_keys,
            "maze_size": tuple(self.maze_dims)
        }

    def _render_game(self):
        maze_h, maze_w = self.maze.shape
        cell_w = self.WIDTH / maze_w
        cell_h = self.HEIGHT / maze_h
        
        # --- Draw Maze ---
        for r in range(maze_h):
            for c in range(maze_w):
                color = self.COLOR_WALL if self.maze[r, c] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, (c * cell_w, r * cell_h, math.ceil(cell_w), math.ceil(cell_h)))
        
        # --- Draw Exit ---
        exit_color = self.COLOR_EXIT_UNLOCKED if self.exit_unlocked else self.COLOR_EXIT_LOCKED
        exit_px, exit_py = self.exit_pos * np.array([cell_w, cell_h])
        pygame.draw.rect(self.screen, exit_color, (exit_px, exit_py, cell_w, cell_h))

        # --- Draw Keys ---
        key_radius = int(min(cell_w, cell_h) * 0.3)
        glow_radius = int(key_radius * 1.5)
        for key_pos in self.key_positions:
            key_px, key_py = (key_pos + 0.5) * np.array([cell_w, cell_h])
            pygame.gfxdraw.filled_circle(self.screen, int(key_px), int(key_py), glow_radius, (*self.COLOR_KEY_GLOW, 80))
            pygame.gfxdraw.filled_circle(self.screen, int(key_px), int(key_py), key_radius, self.COLOR_KEY)
            pygame.gfxdraw.aacircle(self.screen, int(key_px), int(key_py), key_radius, self.COLOR_KEY)

        # --- Draw Player ---
        player_radius = int(min(cell_w, cell_h) * 0.35)
        player_glow_radius = int(player_radius * 1.7)
        player_px, player_py = (self.player_pos + 0.5) * np.array([cell_w, cell_h])
        pygame.gfxdraw.filled_circle(self.screen, int(player_px), int(player_py), player_glow_radius, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, int(player_px), int(player_py), player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(player_px), int(player_py), player_radius, self.COLOR_PLAYER_GLOW)
        
    def _render_ui(self):
        # --- UI Background ---
        ui_bar = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # --- Keys Text ---
        key_text = f"Keys: {self.keys_collected_count} / {self.num_keys}"
        key_surf = self.font_small.render(key_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(key_surf, (10, 5))

        # --- Score Text ---
        score_text = f"Score: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, 15))
        self.screen.blit(score_surf, score_rect)

        # --- Time Text ---
        time_text = f"Time: {self.steps_remaining}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        time_rect = time_surf.get_rect(right=self.WIDTH - 10, top=5)
        self.screen.blit(time_surf, time_rect)

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE!" if self.last_game_won else "TIME UP!"
            color = self.COLOR_EXIT_UNLOCKED if self.last_game_won else self.COLOR_EXIT_LOCKED
            
            end_text_surf = self.font_large.render(msg, True, color)
            end_text_rect = end_text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a display window
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # The game is turn-based, so we only step on key presses
                if not terminated:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    
                    if action[0] != 0: # Only step if a move key was pressed
                        obs, reward, terminated, _, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                
                # Allow resetting the game
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    print("--- GAME RESET ---")

        # Render the environment's observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for the display window

    env.close()