
# Generated: 2025-08-28T05:24:15.376148
# Source Brief: brief_02599.md
# Brief Index: 2599

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Collect all 5 keys while avoiding the shadow."
    )

    game_description = (
        "Evade a lurking shadow in a procedurally generated isometric maze to collect 5 keys within 600 turns."
    )

    auto_advance = False

    # --- Colors and Constants ---
    COLOR_BG = (15, 18, 32)
    COLOR_WALL_TOP = (40, 45, 70)
    COLOR_WALL_SIDE_L = (30, 35, 60)
    COLOR_WALL_SIDE_R = (25, 30, 50)
    COLOR_FLOOR = (20, 22, 42)
    COLOR_PLAYER = (230, 230, 255)
    COLOR_PLAYER_GLOW = (180, 180, 255)
    COLOR_SHADOW = (5, 5, 10)
    COLOR_KEY = (255, 220, 50)
    COLOR_KEY_GLOW = (255, 180, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    MAZE_WIDTH = 16
    MAZE_HEIGHT = 16
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    TILE_DEPTH = 20
    MAX_STEPS = 600
    NUM_KEYS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_large = pygame.font.SysFont("Arial", 24)
            self.font_small = pygame.font.SysFont("Arial", 16)

        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.maze = []
        self.player_pos = (0, 0)
        self.shadow_pos = (0, 0)
        self.key_locations = []
        self.keys_collected = 0
        self.flicker_state = 0
        self.win_message = ""
        self.random_flickers = []

        self.reset()
        self.validate_implementation()

    def _iso_transform(self, x, y):
        """Converts grid coordinates to screen coordinates."""
        screen_x = (self.screen_width / 2) + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = (self.screen_height / 4) + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _generate_maze(self):
        """Generates a maze using Randomized DFS, ensuring full connectivity."""
        maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.uint8)  # 1 for wall, 0 for path
        stack = [(0, 0)]
        visited = set([(0, 0)])
        maze[0, 0] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = random.choice(neighbors)
                # Carve path
                maze[nx, ny] = 0
                maze[x + (nx - x) // 2, y + (ny - y) // 2] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_path_a_star(self, start, end):
        """A* pathfinding for the shadow."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = { (x,y): float('inf') for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT) }
        g_score[start] = 0
        f_score = { (x,y): float('inf') for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT) }
        f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.MAZE_WIDTH and 0 <= neighbor[1] < self.MAZE_HEIGHT):
                    continue
                if self.maze[neighbor[0], neighbor[1]] == 1:
                    continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.keys_collected = 0
        self.win_message = ""

        self.maze = self._generate_maze()
        
        open_tiles = [(x, y) for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT) if self.maze[x, y] == 0]
        random.shuffle(open_tiles)
        
        self.player_pos = open_tiles.pop()
        
        # Place shadow at least 5 tiles away
        while True:
            self.shadow_pos = open_tiles.pop()
            dist = abs(self.player_pos[0] - self.shadow_pos[0]) + abs(self.player_pos[1] - self.shadow_pos[1])
            if dist >= 5:
                break
            open_tiles.insert(0, self.shadow_pos) # Put it back if too close
        
        self.key_locations = [open_tiles.pop() for _ in range(self.NUM_KEYS)]

        self.random_flickers = [(random.uniform(0.5, 1.5), random.uniform(0.1, 0.4)) for _ in range(10)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        old_player_pos = self.player_pos
        
        # --- Player Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # North
        elif movement == 2: dy = 1   # South
        elif movement == 3: dx = -1  # West
        elif movement == 4: dx = 1   # East
        
        if movement != 0:
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if 0 <= new_pos[0] < self.MAZE_WIDTH and 0 <= new_pos[1] < self.MAZE_HEIGHT and self.maze[new_pos[0], new_pos[1]] == 0:
                self.player_pos = new_pos

        # --- Reward Calculation ---
        dist_to_shadow_before = abs(old_player_pos[0] - self.shadow_pos[0]) + abs(old_player_pos[1] - self.shadow_pos[1])
        dist_to_shadow_after = abs(self.player_pos[0] - self.shadow_pos[0]) + abs(self.player_pos[1] - self.shadow_pos[1])
        if dist_to_shadow_after < dist_to_shadow_before:
            reward -= 0.2

        if self.key_locations:
            dist_to_keys_before = min([abs(old_player_pos[0] - kx) + abs(old_player_pos[1] - ky) for kx, ky in self.key_locations])
            dist_to_keys_after = min([abs(self.player_pos[0] - kx) + abs(self.player_pos[1] - ky) for kx, ky in self.key_locations])
            if dist_to_keys_after < dist_to_keys_before:
                reward += 0.1

        # --- Key Collection ---
        if self.player_pos in self.key_locations:
            self.key_locations.remove(self.player_pos)
            self.keys_collected += 1
            reward += 10
            # sfx: key_collect.wav

        # --- Shadow Movement ---
        path_to_player = self._find_path_a_star(self.shadow_pos, self.player_pos)
        if path_to_player:
            # Move up to 2 steps
            move_dist = min(2, len(path_to_player))
            self.shadow_pos = path_to_player[move_dist - 1]
            # sfx: shadow_step.wav

        # --- Update State and Check Termination ---
        self.steps += 1
        
        if self.player_pos == self.shadow_pos:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_message = "CAUGHT"
            # sfx: game_over_caught.wav
        elif self.keys_collected == self.NUM_KEYS:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "ESCAPE SUCCESSFUL"
            # sfx: game_win.wav
        elif self.steps >= self.MAX_STEPS:
            reward -= 10
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"
            # sfx: game_over_timeout.wav

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_tile(self, x, y, base_color_top, base_color_l, base_color_r):
        """Renders a single isometric tile cube."""
        sx, sy = self._iso_transform(x, y)
        
        # Points for the top diamond
        top_points = [
            (sx, sy - self.TILE_HEIGHT / 2),
            (sx + self.TILE_WIDTH / 2, sy),
            (sx, sy + self.TILE_HEIGHT / 2),
            (sx - self.TILE_WIDTH / 2, sy)
        ]
        
        # Points for the left side
        left_points = [
            (sx - self.TILE_WIDTH / 2, sy),
            (sx, sy + self.TILE_HEIGHT / 2),
            (sx, sy + self.TILE_HEIGHT / 2 + self.TILE_DEPTH),
            (sx - self.TILE_WIDTH / 2, sy + self.TILE_DEPTH)
        ]
        
        # Points for the right side
        right_points = [
            (sx + self.TILE_WIDTH / 2, sy),
            (sx, sy + self.TILE_HEIGHT / 2),
            (sx, sy + self.TILE_HEIGHT / 2 + self.TILE_DEPTH),
            (sx + self.TILE_WIDTH / 2, sy + self.TILE_DEPTH)
        ]

        pygame.gfxdraw.filled_polygon(self.screen, left_points, base_color_l)
        pygame.gfxdraw.filled_polygon(self.screen, right_points, base_color_r)
        pygame.gfxdraw.filled_polygon(self.screen, top_points, base_color_top)
        
        # Outlines for definition
        pygame.gfxdraw.aapolygon(self.screen, top_points, base_color_top)
        pygame.gfxdraw.aapolygon(self.screen, left_points, base_color_l)
        pygame.gfxdraw.aapolygon(self.screen, right_points, base_color_r)

    def _render_game(self):
        """Renders all game elements in order."""
        self.flicker_state += 0.1

        # Render maze structure
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[x, y] == 1:
                    self._render_tile(x, y, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE_L, self.COLOR_WALL_SIDE_R)
                else:
                    sx, sy = self._iso_transform(x,y)
                    floor_points = [
                        (sx, sy + self.TILE_DEPTH - self.TILE_HEIGHT / 2),
                        (sx + self.TILE_WIDTH / 2, sy + self.TILE_DEPTH),
                        (sx, sy + self.TILE_DEPTH + self.TILE_HEIGHT / 2),
                        (sx - self.TILE_WIDTH / 2, sy + self.TILE_DEPTH)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)

        # Render keys
        for kx, ky in self.key_locations:
            sx, sy = self._iso_transform(kx, ky)
            sy += self.TILE_DEPTH - 5
            pulse = (math.sin(self.flicker_state * 2) + 1) / 2
            
            # Glow
            glow_radius = int(8 + pulse * 4)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, (*self.COLOR_KEY_GLOW, 50))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, glow_radius, (*self.COLOR_KEY_GLOW, 80))
            
            # Key glyph
            key_radius = 5
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, key_radius, self.COLOR_KEY)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, key_radius, self.COLOR_KEY)

        # Render shadow
        sx, sy = self._iso_transform(self.shadow_pos[0], self.shadow_pos[1])
        sy += self.TILE_DEPTH
        for i in range(5):
            offset_x = math.sin(self.flicker_state * self.random_flickers[i][0] + i) * 5
            offset_y = math.cos(self.flicker_state * self.random_flickers[i][1] + i) * 3
            radius = int(10 + math.sin(self.flicker_state + i) * 3)
            pygame.gfxdraw.filled_circle(self.screen, int(sx + offset_x), int(sy + offset_y), radius, (*self.COLOR_SHADOW, 100))

        # Render player
        sx, sy = self._iso_transform(self.player_pos[0], self.player_pos[1])
        sy += self.TILE_DEPTH - 8 # Elevate slightly
        
        # Glow
        glow_radius = int(10 + (math.sin(self.flicker_state * 1.5) + 1) * 2)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, (*self.COLOR_PLAYER_GLOW, 40))
        pygame.gfxdraw.aacircle(self.screen, sx, sy, glow_radius, (*self.COLOR_PLAYER_GLOW, 60))
        
        # Player core
        player_radius = 6
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        """Renders the UI text overlay."""
        # Key count
        key_text_str = f"Keys: {self.keys_collected} / {self.NUM_KEYS}"
        key_surf_fg = self.font_large.render(key_text_str, True, self.COLOR_TEXT)
        key_surf_bg = self.font_large.render(key_text_str, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(key_surf_bg, (11, 11))
        self.screen.blit(key_surf_fg, (10, 10))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text_str = f"Turns Left: {time_left}"
        time_surf_fg = self.font_large.render(time_text_str, True, self.COLOR_TEXT)
        time_surf_bg = self.font_large.render(time_text_str, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(time_surf_bg, (self.screen_width - time_surf_fg.get_width() - 9, 11))
        self.screen.blit(time_surf_fg, (self.screen_width - time_surf_fg.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_surf_fg = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            win_surf_bg = self.font_large.render(self.win_message, True, self.COLOR_TEXT_SHADOW)
            
            win_rect = win_surf_fg.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(win_surf_bg, (win_rect.x + 2, win_rect.y + 2))
            self.screen.blit(win_surf_fg, win_rect)

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
            "keys_collected": self.keys_collected,
            "player_pos": self.player_pos,
            "shadow_pos": self.shadow_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display and is for testing/demonstration.
    # To run headless for training, this block can be removed.
    try:
        # Re-initialize pygame with a display
        pygame.display.init()
        pygame.font.init() # re-init after display
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption(env.game_description)
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            action = [0, 0, 0] # Default: no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if action[0] != 0: # Only step if a move key is pressed
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # If game over, wait for a moment before closing
            if terminated:
                pygame.time.wait(2000)

    finally:
        env.close()