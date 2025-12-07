import gymnasium as gym
import os
import pygame
import os
import pygame


# This is for running on a server without a display
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
from collections import deque


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a procedurally generated labyrinth to find the exit while being hunted by a relentless Minotaur. "
        "Use portals and time-slowing abilities to survive."
    )
    user_guide = "Controls: Use arrow keys to move. Press space to place a portal and shift to activate a time-slowing ability."
    auto_advance = False  # Game is turn-based; state advances only on action.

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.INITIAL_MAZE_DIM = 15
        self.DIFFICULTY_INTERVAL = 5  # Labyrinth size increases every 5 wins

        # --- Colors ---
        self.COLOR_BG = (10, 20, 35)
        self.COLOR_BG_SLOW = (10, 35, 50)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_FLOOR = (20, 30, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_MINOTAUR = (255, 50, 50)
        self.COLOR_MINOTAUR_GLOW = (255, 120, 120)
        self.COLOR_EXIT = (255, 215, 0)
        self.COLOR_EXIT_GLOW = (255, 235, 100)
        self.COLOR_PORTAL_1 = (180, 0, 255)
        self.COLOR_PORTAL_2 = (0, 255, 180)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_INFO_BG = (0, 0, 0, 128)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 16)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.maze_dim = self.INITIAL_MAZE_DIM
        self.wins_in_a_row = 0

        self.maze = None
        self.player_pos = None
        self.minotaur_pos = None
        self.exit_pos = None

        self.player_render_pos = None
        self.minotaur_render_pos = None

        self.portals = []
        self.portal_charges = 0
        self.time_slow_turns_left = 0

        self.portal_particles = []
        self.time_wave_phase = 0
        
        self.np_random = None

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)  # 1 = wall
        stack = deque()

        start_x = self.np_random.integers(0, width // 2) * 2 + 1
        start_y = self.np_random.integers(0, height // 2) * 2 + 1
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                maze[ny, nx] = 0
                maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Difficulty Scaling ---
        if self.game_over and self.player_pos == self.exit_pos:
            self.wins_in_a_row += 1
            if self.wins_in_a_row % self.DIFFICULTY_INTERVAL == 0:
                self.maze_dim = min(35, self.maze_dim + 2)
        elif self.game_over:  # Reset on loss
            self.wins_in_a_row = 0
            self.maze_dim = self.INITIAL_MAZE_DIM

        # --- State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)

        # Place entities
        floor_tiles = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        self.player_pos = tuple(floor_tiles.pop())
        self.exit_pos = tuple(floor_tiles.pop())

        # Ensure Minotaur starts far from player
        while True:
            if not floor_tiles: # Handle small mazes where no tiles are left
                all_floor = np.argwhere(self.maze == 0).tolist()
                possible_mino_pos = [tuple(t) for t in all_floor if tuple(t) != self.player_pos and tuple(t) != self.exit_pos]
                self.minotaur_pos = possible_mino_pos[self.np_random.integers(len(possible_mino_pos))]
                break
            
            self.minotaur_pos = tuple(floor_tiles.pop())
            if self._get_manhattan_distance(self.player_pos, self.minotaur_pos) > self.maze_dim // 2:
                break

        # Rendering state
        self.tile_size = min(self.WIDTH // self.maze_dim, self.HEIGHT // self.maze_dim)
        self.grid_offset_x = (self.WIDTH - self.maze_dim * self.tile_size) / 2
        self.grid_offset_y = (self.HEIGHT - self.maze_dim * self.tile_size) / 2

        self.player_render_pos = self._grid_to_pixel(self.player_pos)
        self.minotaur_render_pos = self._grid_to_pixel(self.minotaur_pos)

        # Gameplay state
        self.portals = []
        self.portal_charges = 3
        self.time_slow_turns_left = 0
        self.portal_particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.game_over and not (self.steps >= self.MAX_STEPS), self.steps >= self.MAX_STEPS, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        prev_player_pos = self.player_pos
        prev_dist_to_exit = self._get_manhattan_distance(prev_player_pos, self.exit_pos)

        # --- 1. Handle Player Actions ---
        # Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self.maze[new_pos[1], new_pos[0]] == 0:
                self.player_pos = new_pos

        # Activate Portal (Shift) - higher priority
        if shift_pressed and self.time_slow_turns_left == 0 and self.portals:
            # Find nearest portal
            self.portals.sort(key=lambda p: self._get_manhattan_distance(self.player_pos, p))
            if self._get_manhattan_distance(self.player_pos, self.portals[0]) < self.maze_dim:  # Activate any portal
                self.time_slow_turns_left = 3
                if self._get_manhattan_distance(self.player_pos, self.minotaur_pos) <= 4:
                    reward += 5.0  # Reward for tactical activation

        # Place Portal (Space)
        elif space_pressed:
            can_place = self.player_pos not in self.portals and self.maze[self.player_pos[1], self.player_pos[0]] == 0
            if self.portal_charges > 0 and can_place:
                self.portals.append(self.player_pos)
                self.portal_charges -= 1
            else:
                reward -= 1.0  # Penalty for failed placement

        # --- 2. Update Game World ---
        if self.time_slow_turns_left > 0:
            self.time_slow_turns_left -= 1
        else:
            # Minotaur moves 2 steps
            for _ in range(2):
                path = self._a_star_path(self.minotaur_pos, self.player_pos)
                if path and len(path) > 1:
                    self.minotaur_pos = path[1]
                else:  # No path or already at player
                    break

        # --- 3. Calculate Rewards ---
        new_dist_to_exit = self._get_manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < prev_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > prev_dist_to_exit:
            reward -= 0.1

        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        if self.player_pos == self.minotaur_pos:
            reward -= 100.0
            terminated = True
        elif self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # --- Update render positions for smooth animation ---
        lerp_factor = 0.25
        target_player_pixel = self._grid_to_pixel(self.player_pos)
        self.player_render_pos = (
            self.player_render_pos[0] + (target_player_pixel[0] - self.player_render_pos[0]) * lerp_factor,
            self.player_render_pos[1] + (target_player_pixel[1] - self.player_render_pos[1]) * lerp_factor,
        )
        target_minotaur_pixel = self._grid_to_pixel(self.minotaur_pos)
        self.minotaur_render_pos = (
            self.minotaur_render_pos[0] + (target_minotaur_pixel[0] - self.minotaur_render_pos[0]) * lerp_factor,
            self.minotaur_render_pos[1] + (target_minotaur_pixel[1] - self.minotaur_render_pos[1]) * lerp_factor,
        )

        # --- Main Rendering ---
        bg_color = self.COLOR_BG_SLOW if self.time_slow_turns_left > 0 else self.COLOR_BG
        self.screen.fill(bg_color)

        self._render_maze()
        self._render_portals()
        self._render_exit()
        self._render_minotaur()
        self._render_player()

        if self.time_slow_turns_left > 0:
            self._render_time_warp_effect()

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_maze(self):
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.tile_size,
                    self.grid_offset_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

    def _render_exit(self):
        pos = self._grid_to_pixel(self.exit_pos)
        radius = self.tile_size // 3
        self._draw_glow_circle(self.screen, pos, radius, self.COLOR_EXIT, self.COLOR_EXIT_GLOW)

    def _render_portals(self):
        self.portal_particles = [p for p in self.portal_particles if p[3] > 0]
        for i, portal_pos in enumerate(self.portals):
            px, py = self._grid_to_pixel(portal_pos)
            radius = int(self.tile_size * 0.4)
            angle = (pygame.time.get_ticks() + i * 1000) * 0.005

            # Swirling effect
            for j in range(5):
                offset_angle = angle + j * (2 * math.pi / 5)
                ox = math.cos(offset_angle) * radius * 0.5
                oy = math.sin(offset_angle) * radius * 0.5
                color = self.COLOR_PORTAL_1 if j % 2 == 0 else self.COLOR_PORTAL_2
                pygame.gfxdraw.aacircle(self.screen, int(px + ox), int(py + oy), int(radius * 0.3), color)

            # Add particles
            if self.np_random.random() < 0.3:
                p_angle = self.np_random.uniform(0, 2 * math.pi)
                p_vel = (math.cos(p_angle), math.sin(p_angle))
                p_color_idx = self.np_random.integers(2)
                p_color = [self.COLOR_PORTAL_1, self.COLOR_PORTAL_2][p_color_idx]
                self.portal_particles.append([(px, py), p_vel, p_color, 30])  # pos, vel, color, life

        for p in self.portal_particles:
            p[0] = (p[0][0] + p[1][0], p[0][1] + p[1][1])
            p[3] -= 1
            alpha = max(0, min(255, int(255 * (p[3] / 30))))
            pygame.gfxdraw.pixel(self.screen, int(p[0][0]), int(p[0][1]), (*p[2], alpha))

    def _render_minotaur(self):
        pos = self.minotaur_render_pos
        radius = self.tile_size // 2 * 0.8
        self._draw_glow_circle(self.screen, pos, int(radius * 1.5), self.COLOR_MINOTAUR, self.COLOR_MINOTAUR_GLOW)

        points = []
        for i in range(3):
            angle = math.radians(i * 120 + 90)
            points.append((pos[0] + radius * math.cos(angle), pos[1] - radius * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINOTAUR)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINOTAUR)

    def _render_player(self):
        pos = self.player_render_pos
        radius = self.tile_size // 3
        self._draw_glow_circle(self.screen, pos, radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_time_warp_effect(self):
        warp_surface = self.screen.copy()
        warp_surface.set_colorkey(self.COLOR_BG_SLOW)
        self.time_wave_phase += 0.5
        for y in range(self.HEIGHT):
            offset = int(math.sin(y * 0.05 + self.time_wave_phase) * 3)
            self.screen.blit(warp_surface, (offset, y), (0, y, self.WIDTH, 1))

    def _render_ui(self):
        # Info Panel Background
        info_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        info_panel.fill(self.COLOR_INFO_BG)
        self.screen.blit(info_panel, (0, self.HEIGHT - 40))

        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.HEIGHT - 32))

        # Portal Charges
        charges_text = self.font_main.render("Portals:", True, self.COLOR_TEXT)
        self.screen.blit(charges_text, (self.WIDTH - 180, self.HEIGHT - 32))
        for i in range(3):
            pos = (self.WIDTH - 80 + i * 25, self.HEIGHT - 20)
            color = self.COLOR_PORTAL_1 if i < self.portal_charges else (50, 50, 50)
            pygame.draw.circle(self.screen, color, pos, 8)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_TEXT)

        # Time State
        if self.time_slow_turns_left > 0:
            time_text = self.font_info.render(f"TIME SLOW ({self.time_slow_turns_left})", True, self.COLOR_PLAYER_GLOW)
            self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, self.HEIGHT - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "minotaur_pos": self.minotaur_pos,
            "exit_pos": self.exit_pos,
            "portal_charges": self.portal_charges,
            "time_slow": self.time_slow_turns_left > 0,
        }

    # --- Helper Functions ---
    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        px = self.grid_offset_x + (x + 0.5) * self.tile_size
        py = self.grid_offset_y + (y + 0.5) * self.tile_size
        return (px, py)

    def _get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _a_star_path(self, start, end):
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._get_manhattan_distance(start, end)}

        while open_set:
            current = min(open_set, key=lambda o: f_score.get(o, float("inf")))
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            open_set.remove(current)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.maze_dim and 0 <= neighbor[1] < self.maze_dim):
                    continue
                if self.maze[neighbor[1], neighbor[0]] == 1:
                    continue

                tentative_g_score = g_score.get(current, float("inf")) + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        return None

    def _draw_glow_circle(self, surface, pos, radius, color, glow_color):
        pos = (int(pos[0]), int(pos[1]))
        max_glow_radius = int(radius * 2.5)
        for i in range(max_glow_radius, radius, -1):
            alpha = 50 * (1 - (i - radius) / (max_glow_radius - radius)) ** 2
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], i, (*glow_color, int(alpha)))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To run with a display, comment out the os.environ line at the top
    # or run this script with SDL_VIDEODRIVER unset.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False

    # --- Pygame setup for manual play ---
    pygame.display.set_caption("Labyrinth of Chronos - Manual Test")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    action = [0, 0, 0]  # No-op, no space, no shift

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Movement
        action[0] = 0  # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4

        # Place Portal
        action[1] = 1 if keys[pygame.K_SPACE] else 0

        # Activate Portal
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if GameEnv.auto_advance:
            clock.tick(env.metadata["render_fps"])
        else:  # For turn-based, a small delay is good for human playability
            pygame.time.wait(50)

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")