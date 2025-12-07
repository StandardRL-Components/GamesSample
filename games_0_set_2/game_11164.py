import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Navigate a priestess through a procedurally generated labyrinth to find the exit, while avoiding a patrolling minotaur. "
        "Use portals strategically to gain an advantage."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move. Press space while on a portal to activate or deactivate the portal network."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.ANIM_DURATION = 200 # ms for tile-to-tile animation
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PATH = (80, 80, 100)
        self.COLOR_PRIESTESS = (255, 255, 255)
        self.COLOR_MINOTAUR = (200, 50, 50)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_PORTAL_INACTIVE = (50, 50, 150)
        self.COLOR_PORTAL_ACTIVE = (100, 150, 255)
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font = pygame.font.Font(None, 24)
        
        # --- Game State Initialization ---
        self.level = 0
        self._initialize_state_variables()
        # self.reset() is called by the environment wrapper, no need to call it here.

    def _initialize_state_variables(self):
        # These are all reset in reset() but defined here to satisfy linters
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward = 0
        
        self.grid_w = 0
        self.grid_h = 0
        self.tile_w = 0
        self.tile_h = 0
        
        self.labyrinth = None
        self.priestess_pos = (0, 0)
        self.minotaur_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.portal_locs = []
        self.portals_active = False
        
        self.minotaur_path = []
        self.minotaur_path_index = 0
        
        # For smooth animation
        self.priestess_vis_pos = [0, 0]
        self.priestess_anim_start = [0, 0]
        self.priestess_last_move_dir = (0, 1) # Down
        self.minotaur_vis_pos = [0, 0]
        self.minotaur_anim_start = [0, 0]
        self.anim_timer = 0

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_state_variables()
        self.game_over = False
        
        # --- Difficulty Scaling ---
        size_multiplier = 1.0 + self.level * 0.1
        self.grid_w = int(16 * size_multiplier)
        self.grid_h = int(10 * size_multiplier)
        minotaur_cycle_len = max(10, 30 - (self.level // 3))

        self.tile_w = self.WIDTH / self.grid_w
        self.tile_h = self.HEIGHT / self.grid_h

        # --- Generate Labyrinth ---
        self.labyrinth = self._generate_labyrinth(self.grid_w, self.grid_h)
        
        open_cells = [(x, y) for y in range(self.grid_h) for x in range(self.grid_w) if self.labyrinth[y][x] == 0]
        random.shuffle(open_cells)

        self.priestess_pos = open_cells.pop()
        self.exit_pos = open_cells.pop()
        self.portal_locs = [open_cells.pop(), open_cells.pop()]
        self.portals_active = False

        # --- Minotaur Setup ---
        self.minotaur_path = self._generate_minotaur_path(minotaur_cycle_len)
        self.minotaur_path_index = self.np_random.integers(0, len(self.minotaur_path))
        self.minotaur_pos = self.minotaur_path[self.minotaur_path_index]
        
        # Ensure priestess and minotaur don't start on the same tile
        while self.minotaur_pos == self.priestess_pos and open_cells:
            self.priestess_pos = open_cells.pop()
        
        # --- Reset Visuals ---
        self.priestess_vis_pos = list(self._grid_to_pixel(self.priestess_pos))
        self.priestess_anim_start = self.priestess_vis_pos.copy()
        self.minotaur_vis_pos = list(self._grid_to_pixel(self.minotaur_pos))
        self.minotaur_anim_start = self.minotaur_vis_pos.copy()
        self.anim_timer = self.ANIM_DURATION # Start fully animated
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.anim_timer = 0
        self.priestess_anim_start = self.priestess_vis_pos.copy()
        self.minotaur_anim_start = self.minotaur_vis_pos.copy()

        movement, space_held, _ = action
        space_held = space_held == 1

        # --- Store state for reward calculation ---
        prev_dist_to_minotaur = self._manhattan_distance(self.priestess_pos, self.minotaur_pos)
        
        # --- Player Movement ---
        px, py = self.priestess_pos
        move_dir = (0, 0)
        if movement == 1: move_dir = (0, -1) # Up
        elif movement == 2: move_dir = (0, 1)  # Down
        elif movement == 3: move_dir = (-1, 0) # Left
        elif movement == 4: move_dir = (1, 0)  # Right

        if movement != 0:
            self.priestess_last_move_dir = move_dir

        nx, ny = px + move_dir[0], py + move_dir[1]
        if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and self.labyrinth[ny][nx] == 0:
            self.priestess_pos = (nx, ny)

        # --- Portal Teleportation (Passive) ---
        if self.portals_active and self.priestess_pos in self.portal_locs:
            # Sound: Portal teleport whoosh
            target_portal = self.portal_locs[1] if self.priestess_pos == self.portal_locs[0] else self.portal_locs[0]
            self.priestess_pos = target_portal
            self._create_particles(self._grid_to_pixel(self.priestess_pos), self.COLOR_PORTAL_ACTIVE, 30, 5)

        # --- Minotaur Movement ---
        self.minotaur_path_index = (self.minotaur_path_index + 1) % len(self.minotaur_path)
        self.minotaur_pos = self.minotaur_path[self.minotaur_path_index]

        # --- Reward Calculation ---
        reward = self._calculate_reward(prev_dist_to_minotaur, space_held)
        self.score += reward
        self.last_reward = reward

        # --- Portal Activation (Action) ---
        if space_held:
            if self.priestess_pos in self.portal_locs:
                self.portals_active = not self.portals_active
                # Sound: Portal activation/deactivation hum
                self._create_particles(self._grid_to_pixel(self.priestess_pos), self.COLOR_PORTAL_ACTIVE, 20, 3)

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.priestess_pos == self.exit_pos:
                self.level += 1 # Increase difficulty for next game

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_reward(self, prev_dist_to_minotaur, space_held):
        reward = -0.01 # Small penalty for each step to encourage efficiency
        
        # 1. Distance from Minotaur
        current_dist_to_minotaur = self._manhattan_distance(self.priestess_pos, self.minotaur_pos)
        if current_dist_to_minotaur > prev_dist_to_minotaur:
            reward += 0.1
        elif current_dist_to_minotaur < prev_dist_to_minotaur:
            reward -= 0.1

        # 2. Portal Activation
        if space_held and self.priestess_pos in self.portal_locs:
            path_to_exit = self._find_path(self.priestess_pos, self.exit_pos)
            dist_before = len(path_to_exit) if path_to_exit else float('inf')
            
            # Simulate teleport
            other_portal = self.portal_locs[1] if self.priestess_pos == self.portal_locs[0] else self.portal_locs[0]
            path_after_teleport = self._find_path(other_portal, self.exit_pos)
            dist_after = len(path_after_teleport) if path_after_teleport else float('inf')

            if dist_after < dist_before:
                reward += 5.0 # Good portal
            else:
                reward -= 1.0 # Bad portal

        # 3. Terminal states
        if self.priestess_pos == self.minotaur_pos:
            return -100.0
        if self.priestess_pos == self.exit_pos:
            return 100.0
        
        return reward

    def _check_termination(self):
        if self.priestess_pos == self.minotaur_pos:
            return True
        if self.priestess_pos == self.exit_pos:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def render(self):
        # This is for human playback, not training.
        # It relies on the _get_observation to do the drawing.
        self._get_observation()
        # The following is needed for human rendering
        if self.metadata["render_modes"][0] != "rgb_array":
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return pygame.surfarray.array3d(pygame.display.get_surface()) if "human" in self.metadata["render_modes"] else self._get_observation()

    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _grid_to_pixel(self, pos):
        x, y = pos
        return (x * self.tile_w + self.tile_w / 2, y * self.tile_h + self.tile_h / 2)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _lerp(self, start, end, t):
        return start + (end - start) * t

    def _generate_labyrinth(self, w, h):
        grid = np.ones((h, w), dtype=np.uint8) # 1 for wall, 0 for path
        stack = deque()
        
        start_x, start_y = (self.np_random.integers(0, w // 2) * 2, self.np_random.integers(0, h // 2) * 2)
        grid[start_y][start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                grid[ny][nx] = 0
                grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid

    def _find_path(self, start, end):
        if not (0 <= start[0] < self.grid_w and 0 <= start[1] < self.grid_h and self.labyrinth[start[1]][start[0]] == 0): return None
        if not (0 <= end[0] < self.grid_w and 0 <= end[1] < self.grid_h and self.labyrinth[end[1]][end[0]] == 0): return None
        
        open_set = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_w and 0 <= neighbor[1] < self.grid_h and self.labyrinth[neighbor[1]][neighbor[0]] == 0:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current
        return None

    def _generate_minotaur_path(self, length):
        open_cells = [(x, y) for y in range(self.grid_h) for x in range(self.grid_w) if self.labyrinth[y][x] == 0]
        if len(open_cells) < 2:
            return [(0,0)] * length
        p1, p2 = random.sample(open_cells, 2)
        
        path1 = self._find_path(p1, p2)
        path2 = self._find_path(p2, p1)

        if not path1 or not path2: # Fallback if pathfinding fails
            return random.choices(open_cells, k=length)

        loop_path = path1 + path2[1:]
        if not loop_path:
            return random.choices(open_cells, k=length)
            
        final_path = []
        while len(final_path) < length:
            final_path.extend(loop_path)
        return final_path[:length]

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.uniform(10, 20),
                "color": color
            })

    def _draw_glow(self, surface, pos, color, max_radius, layers=5):
        temp_surf = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
        for i in range(layers, 0, -1):
            radius = int(max_radius * (i / layers))
            alpha = int(100 * (1 - (i / layers))**2)
            pygame.gfxdraw.filled_circle(temp_surf, int(max_radius), int(max_radius), radius, (*color, alpha))
        surface.blit(temp_surf, (int(pos[0] - max_radius), int(pos[1] - max_radius)))


    # --- Rendering Methods ---

    def _render_game(self):
        # --- Update animation timer and interpolation factor ---
        self.anim_timer += self.clock.get_time()
        t = min(1.0, self.anim_timer / self.ANIM_DURATION)

        # --- Draw Labyrinth ---
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                rect = (x * self.tile_w, y * self.tile_h, self.tile_w, self.tile_h)
                color = self.COLOR_WALL if self.labyrinth[y][x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # --- Draw Exit ---
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        self._draw_glow(self.screen, (exit_px, exit_py), self.COLOR_EXIT, self.tile_w * 0.8)
        pygame.draw.circle(self.screen, self.COLOR_EXIT, (exit_px, exit_py), self.tile_w * 0.4)
        
        # --- Draw Portals ---
        for i, pos in enumerate(self.portal_locs):
            px, py = self._grid_to_pixel(pos)
            color = self.COLOR_PORTAL_ACTIVE if self.portals_active else self.COLOR_PORTAL_INACTIVE
            radius = self.tile_w * 0.4
            if self.portals_active:
                self._draw_glow(self.screen, (px, py), color, radius * 1.8)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(radius), color)
            
        # --- Interpolate and Draw Minotaur ---
        target_vis_pos = self._grid_to_pixel(self.minotaur_pos)
        self.minotaur_vis_pos = [
            self._lerp(self.minotaur_anim_start[0], target_vis_pos[0], t),
            self._lerp(self.minotaur_anim_start[1], target_vis_pos[1], t)
        ]
        mx, my = self.minotaur_vis_pos
        pulse = 1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.01)
        minotaur_size = self.tile_w * 0.35 * pulse
        self._draw_glow(self.screen, (mx, my), self.COLOR_MINOTAUR, minotaur_size * 2.5)
        pygame.draw.rect(self.screen, self.COLOR_MINOTAUR, (mx - minotaur_size, my - minotaur_size, minotaur_size * 2, minotaur_size * 2))

        # --- Interpolate and Draw Priestess ---
        target_vis_pos = self._grid_to_pixel(self.priestess_pos)
        self.priestess_vis_pos = [
            self._lerp(self.priestess_anim_start[0], target_vis_pos[0], t),
            self._lerp(self.priestess_anim_start[1], target_vis_pos[1], t)
        ]
        px, py = self.priestess_vis_pos
        priestess_size = self.tile_w * 0.3
        self._draw_glow(self.screen, (px, py), self.COLOR_PRIESTESS, priestess_size * 2)

        # Draw priestess as a triangle
        angle = math.atan2(self.priestess_last_move_dir[1], self.priestess_last_move_dir[0])
        p1 = (px + math.cos(angle) * priestess_size, py + math.sin(angle) * priestess_size)
        p2 = (px + math.cos(angle + 2.2) * priestess_size, py + math.sin(angle + 2.2) * priestess_size)
        p3 = (px + math.cos(angle - 2.2) * priestess_size, py + math.sin(angle - 2.2) * priestess_size)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PRIESTESS)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PRIESTESS)

        # --- Update and Draw Particles ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color = (*p['color'], alpha)
                radius = int(p['life'] / 4)
                if radius > 0:
                    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
                    self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        level_text = self.font.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It's important to set the render_mode to "human" for interactive playback
    env = GameEnv(render_mode="human")
    env.metadata["render_modes"] = ["human"] # Force human rendering mode
    pygame.display.set_caption("Labyrinth of the Minotaur")
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrow keys for movement, Space for portals
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        movement_action = 0  # No-op
        space_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move in key_to_movement.items():
            if keys[key]:
                movement_action = move
                break
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            done = False
            continue
        
        if keys[pygame.K_ESCAPE]:
            running = False

        action = [movement_action, space_action, 0] # Shift is unused
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        env.render() # Renders to the display window

    env.close()