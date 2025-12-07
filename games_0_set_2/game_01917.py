
# Generated: 2025-08-28T03:06:28.548644
# Source Brief: brief_01917.md
# Brief Index: 1917

        
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
        "Controls: Use arrow keys to navigate the maze. Reach the green exit before time runs out. "
        "Collect blue checkpoints for points. Press space on a flashing gold tile for a bonus."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze against the clock. Collect checkpoints and find the exit to maximize your score. The maze gets bigger as you succeed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_TIME = 60
        self.INITIAL_MAZE_DIM = 5
        self.DIFFICULTY_INTERVAL = 5

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_PATH = (60, 70, 90)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_CHECKPOINT = (0, 150, 255)
        self.COLOR_BONUS_1 = (255, 220, 50)
        self.COLOR_BONUS_2 = (255, 180, 0)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (persistent across resets) ---
        self.successful_episodes = 0
        self.maze_dim = self.INITIAL_MAZE_DIM

        # --- Game State (reset each episode) ---
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.maze_grid = None
        self.player_pos = None
        self.start_pos = None
        self.exit_pos = None
        self.checkpoints = []
        self.collected_checkpoints = []
        self.bonus_tile = None
        self.particles = []
        self.prev_dist_to_exit = 0
        self.tile_size = 0
        self.offset_x = 0
        self.offset_y = 0

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.successful_episodes > 0 and self.successful_episodes % self.DIFFICULTY_INTERVAL == 0:
            self.maze_dim = min(self.maze_dim + 2, 25) # Cap difficulty
            self.successful_episodes += 1 # Prevent re-triggering until next milestone

        self.steps = 0
        self.score = 0
        self.time_remaining = self.INITIAL_TIME + (self.maze_dim - self.INITIAL_MAZE_DIM) * 5
        self.game_over = False
        self.particles = []

        self.maze_grid = self._generate_maze(self.maze_dim, self.maze_dim)
        self.start_pos = np.array([1, 1])
        self.player_pos = self.start_pos.copy()
        grid_dim = self.maze_dim * 2
        self.exit_pos = np.array([grid_dim - 1, grid_dim - 1])

        self._place_game_elements()
        self.prev_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed = action[0], action[1] == 1
        reward = 0.0

        self.steps += 1
        self.time_remaining -= 1

        if self.bonus_tile:
            self.bonus_tile["lifetime"] -= 1
            if self.bonus_tile["lifetime"] <= 0:
                self.bonus_tile = None # Vanished

        moved = False
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            new_pos = self.player_pos + np.array([dx, dy])

            if self.maze_grid[new_pos[1], new_pos[0]] == 0:
                self.player_pos = new_pos
                moved = True
            else:
                reward -= 1.0 # Bump into wall

        if space_pressed:
            if self.bonus_tile and np.array_equal(self.player_pos, self.bonus_tile["pos"]):
                reward += 20.0
                self.score += 50
                self.time_remaining = min(self.INITIAL_TIME * 2, self.time_remaining + 10)
                self._add_particles(self.player_pos, self.COLOR_BONUS_1, 30)
                self.bonus_tile = None # Collected
            else:
                reward -= 5.0 # Pressed space unnecessarily

        current_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        if current_dist < self.prev_dist_to_exit:
            reward += 0.1
        elif moved and current_dist > self.prev_dist_to_exit:
            reward -= 0.2
        self.prev_dist_to_exit = current_dist

        for i, cp_pos in enumerate(self.checkpoints):
            if i not in self.collected_checkpoints and np.array_equal(self.player_pos, cp_pos):
                self.collected_checkpoints.append(i)
                reward += 5.0
                self.score += 25
                self._add_particles(cp_pos, self.COLOR_CHECKPOINT, 20)
                break

        terminated = False
        if np.array_equal(self.player_pos, self.exit_pos):
            reward += 50.0 + self.time_remaining
            self.score += 100
            terminated = True
            self.game_over = True
            self.successful_episodes += 1
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100.0
            terminated = True
            self.game_over = True

        self._update_particles()
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        padding = 40
        maze_render_w = self.SCREEN_WIDTH - 2 * padding
        maze_render_h = self.SCREEN_HEIGHT - 2 * padding
        grid_h, grid_w = self.maze_grid.shape
        self.tile_size = min(maze_render_w / grid_w, maze_render_h / grid_h)
        self.offset_x = (self.SCREEN_WIDTH - grid_w * self.tile_size) / 2
        self.offset_y = (self.SCREEN_HEIGHT - grid_h * self.tile_size) / 2

        for y in range(grid_h):
            for x in range(grid_w):
                rect = pygame.Rect(self.offset_x + x * self.tile_size, self.offset_y + y * self.tile_size, math.ceil(self.tile_size), math.ceil(self.tile_size))
                color = self.COLOR_WALL if self.maze_grid[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        self._draw_tile_element(self.exit_pos, self.COLOR_EXIT)
        for i, cp_pos in enumerate(self.checkpoints):
            if i not in self.collected_checkpoints:
                self._draw_tile_element(cp_pos, self.COLOR_CHECKPOINT, shape='square')
        
        if self.bonus_tile:
            color = self.COLOR_BONUS_1 if (self.steps // 3) % 2 == 0 else self.COLOR_BONUS_2
            self._draw_tile_element(self.bonus_tile["pos"], color, shape='diamond')

        px, py = self._get_pixel_pos(self.player_pos)
        player_radius = int(self.tile_size * 0.35)
        glow_radius = int(player_radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.filled_circle(self.screen, px, py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, player_radius, self.COLOR_PLAYER)

        for p in self.particles:
            p_rad = int(p['size'] * p['life'])
            if p_rad > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p_rad, p['color'])

    def _render_ui(self):
        self._draw_text(f"SCORE: {self.score}", (self.SCREEN_WIDTH - 10, 10), align="topright")
        self._draw_text(f"TIME: {self.time_remaining}", (10, 10), align="topleft")
        if self.game_over:
            msg = "YOU WIN!" if np.array_equal(self.player_pos, self.exit_pos) else "TIME'S UP!"
            color = self.COLOR_EXIT if np.array_equal(self.player_pos, self.exit_pos) else self.COLOR_PLAYER
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), self.font_big, color, align="center")

    def _generate_maze(self, width, height):
        grid_w, grid_h = width * 2 + 1, height * 2 + 1
        maze = np.ones((grid_h, grid_w), dtype=np.uint8)
        stack = [(1, 1)]
        maze[1, 1] = 0
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < grid_w and 0 < ny < grid_h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                maze[ny, nx] = 0
                maze[(cy + ny) // 2, (cx + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _place_game_elements(self):
        path_tiles = np.argwhere(self.maze_grid == 0)
        valid_tiles = [tuple(pt) for pt in path_tiles if not np.array_equal(pt, self.start_pos) and not np.array_equal(pt, self.exit_pos)]
        self.np_random.shuffle(valid_tiles)
        
        num_checkpoints = self.maze_dim // 2
        self.checkpoints = [np.array(pos) for pos in valid_tiles[:num_checkpoints]]
        self.collected_checkpoints = []
        
        self.bonus_tile = None
        if self.np_random.random() < 0.5 and len(valid_tiles) > num_checkpoints:
            self.bonus_tile = {"pos": np.array(valid_tiles[num_checkpoints]), "lifetime": self.np_random.integers(20, 40)}

    def _add_particles(self, grid_pos, color, count):
        px, py = self._get_pixel_pos(grid_pos)
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({'pos': [px, py], 'vel': [math.cos(angle) * speed, math.sin(angle) * speed], 'life': 1.0, 'size': self.np_random.random() * 3 + 2, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.04
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_pixel_pos(self, grid_pos):
        x = self.offset_x + (grid_pos[0] + 0.5) * self.tile_size
        y = self.offset_y + (grid_pos[1] + 0.5) * self.tile_size
        return int(x), int(y)

    def _draw_tile_element(self, grid_pos, color, shape='rect'):
        cx, cy = self._get_pixel_pos(grid_pos)
        size = int(self.tile_size * 0.7)
        hs = size // 2
        if shape == 'rect' or shape == 'square':
            pygame.draw.rect(self.screen, color, (cx - hs, cy - hs, size, size), border_radius=3)
        elif shape == 'diamond':
            pts = [(cx, cy - hs), (cx + hs, cy), (cx, cy + hs), (cx - hs, cy)]
            pygame.gfxdraw.filled_polygon(self.screen, pts, color)
            pygame.gfxdraw.aapolygon(self.screen, pts, color)

    def _draw_text(self, text, pos, font=None, color=None, align="topleft"):
        font = font or self.font_ui
        color = color or self.COLOR_TEXT
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if align == "topright": rect.topright = pos
        elif align == "center": rect.center = pos
        else: rect.topleft = pos
        self.screen.blit(shadow, (rect.x + 2, rect.y + 2))
        self.screen.blit(surface, rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs_shape = (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert self._get_observation().shape == obs_shape and self._get_observation().dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == obs_shape and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == obs_shape and isinstance(reward, float) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")