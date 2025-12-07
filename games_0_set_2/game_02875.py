import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys (↑↓←→) to navigate the maze."

    # Must be a short, user-facing description of the game:
    game_description = "Navigate a procedurally generated maze to reach the exit before the timer runs out."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # Class attribute for cumulative difficulty across episodes
    cumulative_steps_survived = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 24)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (200, 210, 220)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_WARN = (255, 50, 50)

        # Game parameters
        self.base_maze_width = 31  # Must be odd
        self.base_maze_height = 19  # Must be odd
        self.player_speed = 3.0
        self.max_time = 60.0
        self.fast_win_time = 20.0
        self.fps = 30  # For auto_advance time calculation

        # State variables (initialized in reset)
        self.maze = None
        self.cell_size_w = None
        self.cell_size_h = None
        self.player_pos = None
        self.player_grid_pos = None
        self.exit_pos = None
        self.distance_map = None
        self.last_dist_to_exit = None
        self.timer = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False

        # This is called here to initialize self.np_random, which is needed for validate_implementation
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.steps > 0:
            GameEnv.cumulative_steps_survived += self.steps

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.max_time

        self._generate_maze_and_objects()
        self._calculate_distance_map()

        self.player_grid_pos = self._get_grid_pos(self.player_pos)
        self.last_dist_to_exit = self.distance_map[self.player_grid_pos[1]][self.player_grid_pos[0]]

        return self._get_observation(), self._get_info()

    def _generate_maze_and_objects(self):
        difficulty_factor = 1.0 + 0.05 * (GameEnv.cumulative_steps_survived // 1000)
        maze_w = 2 * int(self.base_maze_width * difficulty_factor / 2) + 1
        maze_h = 2 * int(self.base_maze_height * difficulty_factor / 2) + 1

        self.cell_size_w = self.screen_width / maze_w
        self.cell_size_h = self.screen_height / maze_h

        self.maze = np.ones((maze_h, maze_w), dtype=np.uint8)
        stack = deque()
        # Generate a random odd integer for the starting coordinates
        start_y = 2 * self.np_random.integers(1, maze_h // 2)
        start_x = 2 * self.np_random.integers(1, maze_w // 2)
        self.maze[start_y, start_x] = 0
        stack.append((start_y, start_x))

        while stack:
            cy, cx = stack[-1]
            neighbors = []
            for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 < ny < maze_h - 1 and 0 < nx < maze_w - 1 and self.maze[ny, nx] == 1:
                    neighbors.append((ny, nx))

            if neighbors:
                # Use the environment's seeded RNG for reproducibility
                ny, nx = neighbors[self.np_random.integers(len(neighbors))]
                wy, wx = cy + (ny - cy) // 2, cx + (nx - cx) // 2
                self.maze[wy, wx] = 0
                self.maze[ny, nx] = 0
                stack.append((ny, nx))
            else:
                stack.pop()

        path_cells = np.argwhere(self.maze == 0)
        start_idx, exit_idx = self.np_random.choice(len(path_cells), 2, replace=False)
        start_y, start_x = path_cells[start_idx]
        exit_y, exit_x = path_cells[exit_idx]

        self.player_pos = [start_x * self.cell_size_w + self.cell_size_w / 2,
                           start_y * self.cell_size_h + self.cell_size_h / 2]
        self.exit_pos = (exit_x, exit_y)

    def _calculate_distance_map(self):
        h, w = self.maze.shape
        self.distance_map = np.full((h, w), -1, dtype=int)
        q = deque()

        exit_x, exit_y = self.exit_pos
        self.distance_map[exit_y, exit_x] = 0
        q.append(((exit_y, exit_x), 0))

        while q:
            (cy, cx), dist = q.popleft()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and self.maze[ny, nx] == 0 and self.distance_map[ny, nx] == -1:
                    self.distance_map[ny, nx] = dist + 1
                    q.append(((ny, nx), dist + 1))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]

        vx, vy = 0, 0
        if movement == 1: vy = -self.player_speed
        elif movement == 2: vy = self.player_speed
        elif movement == 3: vx = -self.player_speed
        elif movement == 4: vx = self.player_speed

        self._move_player(vx, vy)

        self.steps += 1
        self.timer -= 1.0 / self.fps

        continuous_reward = self._calculate_continuous_reward()
        terminated, terminal_reward = self._check_termination()

        reward = continuous_reward + terminal_reward
        self.score += reward

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_player(self, vx, vy):
        player_half_w = (self.cell_size_w * 0.6) / 2
        player_half_h = (self.cell_size_h * 0.6) / 2

        # Move X
        if vx != 0:
            new_x = self.player_pos[0] + vx
            grid_y = int(self.player_pos[1] / self.cell_size_h)
            grid_x_to_check = int((new_x + player_half_w * np.sign(vx)) / self.cell_size_w)

            if self.maze[grid_y, grid_x_to_check] == 1:
                if vx > 0:
                    self.player_pos[0] = grid_x_to_check * self.cell_size_w - player_half_w - 0.01
                else:
                    self.player_pos[0] = (grid_x_to_check + 1) * self.cell_size_w + player_half_w + 0.01
            else:
                self.player_pos[0] = new_x

        # Move Y
        if vy != 0:
            new_y = self.player_pos[1] + vy
            grid_x = int(self.player_pos[0] / self.cell_size_w)
            grid_y_to_check = int((new_y + player_half_h * np.sign(vy)) / self.cell_size_h)

            if self.maze[grid_y_to_check, grid_x] == 1:
                if vy > 0:
                    self.player_pos[1] = grid_y_to_check * self.cell_size_h - player_half_h - 0.01
                else:
                    self.player_pos[1] = (grid_y_to_check + 1) * self.cell_size_h + player_half_h + 0.01
            else:
                self.player_pos[1] = new_y

    def _calculate_continuous_reward(self):
        reward = 0
        new_grid_pos = self._get_grid_pos(self.player_pos)

        if new_grid_pos != self.player_grid_pos:
            current_dist = self.distance_map[new_grid_pos[1]][new_grid_pos[0]]
            if current_dist != -1 and self.last_dist_to_exit != -1:
                if current_dist < self.last_dist_to_exit:
                    reward += 0.1
                elif current_dist > self.last_dist_to_exit:
                    reward -= 0.1
            self.last_dist_to_exit = current_dist
            self.player_grid_pos = new_grid_pos
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        if self.player_grid_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            terminal_reward += 10
            if self.max_time - self.timer <= self.fast_win_time:
                terminal_reward += 50
        elif self.timer <= 0:
            terminated = True
            self.game_over = True
            self.timer = 0
        return terminated, terminal_reward

    def _get_grid_pos(self, pos):
        return (int(pos[0] / self.cell_size_w), int(pos[1] / self.cell_size_h))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                if self.maze[r, c] == 1:
                    rect = pygame.Rect(c * self.cell_size_w, r * self.cell_size_h, math.ceil(self.cell_size_w),
                                       math.ceil(self.cell_size_h))
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        exit_rect = pygame.Rect(self.exit_pos[0] * self.cell_size_w, self.exit_pos[1] * self.cell_size_h,
                                self.cell_size_w, self.cell_size_h)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        player_center = (int(self.player_pos[0]), int(self.player_pos[1]))
        player_size_w = int(self.cell_size_w * 0.6)
        player_size_h = int(self.cell_size_h * 0.6)

        glow_size = int(max(player_size_w, player_size_h) * 2.0)
        if glow_size > 1:
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size // 2, glow_size // 2), glow_size // 2)
            self.screen.blit(glow_surf, (player_center[0] - glow_size // 2, player_center[1] - glow_size // 2),
                             special_flags=pygame.BLEND_RGBA_ADD)

        if player_size_w > 0 and player_size_h > 0:
            player_rect = pygame.Rect(0, 0, player_size_w, player_size_h)
            player_rect.center = player_center
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_color = self.COLOR_TEXT_WARN if self.timer < 10 else self.COLOR_TEXT
        timer_text = self.font_ui.render(f"TIME: {max(0, self.timer):.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()