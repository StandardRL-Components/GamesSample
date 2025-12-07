
# Generated: 2025-08-28T03:27:40.633789
# Source Brief: brief_04940.md
# Brief Index: 4940

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down maze puzzle. Find the exit, collect yellow checkpoints for more time and points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 60)
        self.COLOR_EXIT = (50, 255, 50)
        self.COLOR_CHECKPOINT = (255, 220, 50)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Game Constants ---
        self.INITIAL_MAZE_DIM = 11
        self.MAX_MAZE_DIM = 31
        self.INITIAL_TIME = 200
        self.MAX_EPISODE_STEPS = 1000
        self.NUM_CHECKPOINTS = 3
        
        # --- Rewards ---
        self.REWARD_EXIT = 100.0
        self.REWARD_TIMEOUT = -50.0
        self.REWARD_CHECKPOINT = 10.0
        self.REWARD_STEP = -0.1
        self.CHECKPOINT_TIME_BONUS = 50

        # --- Game State (dynamic) ---
        self.current_maze_dim = self.INITIAL_MAZE_DIM
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.checkpoints = None
        self.collected_checkpoints = None
        self.time_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""

        # Initialize a temporary state to pass validation
        self._initialize_first_state()
        self.validate_implementation()

    def _initialize_first_state(self):
        """A helper to set up a valid initial state for validation without a full reset."""
        self.np_random, _ = gym.utils.seeding.np_random()
        self.current_maze_dim = self.INITIAL_MAZE_DIM
        self._generate_and_populate_maze()
        self.time_remaining = self.INITIAL_TIME
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""

    def _generate_maze(self, width, height):
        """Generates a maze using iterative depth-first search (DFS)."""
        w, h = (width // 2) * 2 + 1, (height // 2) * 2 + 1
        maze = np.ones((h, w), dtype=np.uint8) # 1 for wall, 0 for path
        
        start_x, start_y = (1, 1)
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < w - 1 and 0 < ny < h - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                maze[ny, nx] = 0
                maze[(cy + ny) // 2, (cx + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def _generate_and_populate_maze(self):
        """Creates a new maze and places player, exit, and checkpoints."""
        dim = int(self.current_maze_dim)
        self.maze = self._generate_maze(dim, dim)
        
        self.player_pos = [1, 1]
        self.exit_pos = [self.maze.shape[1] - 2, self.maze.shape[0] - 2]
        
        path_cells = np.argwhere(self.maze == 0).tolist()
        
        if list(self.player_pos) in path_cells: path_cells.remove(list(self.player_pos))
        if list(self.exit_pos) in path_cells: path_cells.remove(list(self.exit_pos))
            
        self.checkpoints = []
        if path_cells and self.NUM_CHECKPOINTS > 0:
            num_to_place = min(self.NUM_CHECKPOINTS, len(path_cells))
            indices = self.np_random.choice(len(path_cells), num_to_place, replace=False)
            self.checkpoints = [path_cells[i] for i in indices]
        
        self.collected_checkpoints = [False] * len(self.checkpoints)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'maze_dim' in options:
            self.current_maze_dim = max(self.INITIAL_MAZE_DIM, (options['maze_dim'] // 2) * 2 + 1)

        self._generate_and_populate_maze()
        
        self.time_remaining = self.INITIAL_TIME
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = self.REWARD_STEP
        terminated = False
        
        self.steps += 1
        self.time_remaining -= 1
        
        px, py = self.player_pos
        nx, ny = px, py

        if movement == 1: ny -= 1 # Up
        elif movement == 2: ny += 1 # Down
        elif movement == 3: nx -= 1 # Left
        elif movement == 4: nx += 1 # Right
        
        if self.maze[ny, nx] == 0:
            self.player_pos = [nx, ny]

        for i, chk_pos in enumerate(self.checkpoints):
            if not self.collected_checkpoints[i] and self.player_pos == list(chk_pos):
                self.collected_checkpoints[i] = True
                self.score += self.REWARD_CHECKPOINT
                reward += self.REWARD_CHECKPOINT
                self.time_remaining += self.CHECKPOINT_TIME_BONUS
                # sfx: checkpoint_get.wav
        
        if self.player_pos == self.exit_pos:
            self.score += self.REWARD_EXIT
            reward += self.REWARD_EXIT
            terminated = True
            self.game_over = True
            self.win_message = "SUCCESS!"
            self.current_maze_dim = min(self.MAX_MAZE_DIM, self.current_maze_dim + 2)
            # sfx: level_complete.wav

        if self.time_remaining <= 0:
            reward += self.REWARD_TIMEOUT
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP!"
            # sfx: game_over.wav
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True
            if not self.win_message: self.win_message = "MAX STEPS"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "maze_dim": self.current_maze_dim,
        }

    def _render_ui(self, surface):
        time_text = self.font_ui.render(f"TIME: {self.time_remaining}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        
        surface.blit(time_text, (self.screen_width - time_text.get_width() - 15, 10))
        surface.blit(score_text, (self.screen_width - score_text.get_width() - 15, 35))
        
        if self.game_over:
            color = self.COLOR_EXIT if "SUCCESS" in self.win_message else self.COLOR_PLAYER
            msg_text = self.font_msg.render(self.win_message, True, color)
            text_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            surface.blit(s, bg_rect.topleft)
            surface.blit(msg_text, text_rect)

    def _render_game(self, surface):
        maze_h, maze_w = self.maze.shape
        padding = 20
        render_w, render_h = self.screen_width - 2 * padding, self.screen_height - 2 * padding
        cell_size = min(render_w / maze_w, render_h / maze_h)
        offset_x = (self.screen_width - maze_w * cell_size) / 2
        offset_y = (self.screen_height - maze_h * cell_size) / 2
        
        for y in range(maze_h):
            for x in range(maze_w):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(offset_x + x * cell_size, offset_y + y * cell_size, math.ceil(cell_size), math.ceil(cell_size))
                    pygame.draw.rect(surface, self.COLOR_WALL, rect)
        
        radius = int(cell_size * 0.3)
        for i, chk_pos in enumerate(self.checkpoints):
            if not self.collected_checkpoints[i]:
                x, y = chk_pos
                center_x = int(offset_x + (x + 0.5) * cell_size)
                center_y = int(offset_y + (y + 0.5) * cell_size)
                pygame.draw.circle(surface, self.COLOR_CHECKPOINT, (center_x, center_y), radius)

        exit_x, exit_y = self.exit_pos
        exit_rect = pygame.Rect(offset_x + exit_x * cell_size, offset_y + exit_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(surface, self.COLOR_EXIT, exit_rect.inflate(-cell_size*0.2, -cell_size*0.2))

        player_x, player_y = self.player_pos
        player_center_x = int(offset_x + (player_x + 0.5) * cell_size)
        player_center_y = int(offset_y + (player_y + 0.5) * cell_size)
        player_radius = int(cell_size * 0.35)
        
        glow_radius = int(player_radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        surface.blit(glow_surf, (player_center_x - glow_radius, player_center_y - glow_radius))
        
        pygame.draw.circle(surface, self.COLOR_PLAYER, (player_center_x, player_center_y), player_radius)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game(self.screen)
        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''Call this at the end of __init__ to verify implementation:'''
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    is_interactive = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"

    if not is_interactive:
        print("Running in headless mode.")
        env = GameEnv()
        obs, info = env.reset(seed=42)
        print("Reset successful. Initial info:", info)
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
            if terminated:
                print("Episode finished. Resetting.")
                obs, info = env.reset(seed=i)
        env.close()
    else:
        print("Running in interactive mode. Close the window to quit.")
        env_vis = GameEnv(render_mode="rgb_array")
        obs, info = env_vis.reset()
        
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Maze Runner")
        
        running = True
        while running:
            action = [0, 0, 0]
            move_made = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_r: 
                        print("Manual reset.")
                        obs, info = env_vis.reset()
                    if action[0] != 0: move_made = True

            if move_made:
                obs, reward, terminated, truncated, info = env_vis.step(action)
                if terminated:
                    print(f"Game Over! Score: {info['score']}. Resetting in 3 seconds...")
                    frame = np.transpose(obs, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(3000)
                    obs, info = env_vis.reset()

            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env_vis.clock.tick(30)
        env_vis.close()