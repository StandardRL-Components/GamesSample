import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move. Avoid the guards' red vision cones. "
        "Collect all three artifacts to win."
    )

    game_description = (
        "A top-down stealth game. Steal three artifacts from a procedurally generated "
        "museum without being detected by patrolling guards."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000
        self.NUM_ARTIFACTS = 3
        self.NUM_GUARDS = 3

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_GRID = (30, 35, 40)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150, 50)
        self.COLOR_GUARD = (255, 50, 50)
        self.COLOR_GUARD_VISION = (255, 0, 0, 40)
        self.COLOR_GUARD_PATH = (255, 100, 0, 100)
        self.ARTIFACT_COLORS = [(0, 150, 255), (255, 255, 0), (200, 0, 255)]
        self.COLOR_PEDESTAL = (45, 55, 65)
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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State ---
        self.game_map = None
        self.player_pos = None
        self.guards = []
        self.artifacts = []
        self.artifacts_collected = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._generate_map()
        self._place_entities()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, return the current state without changes.
            # The episode should be reset by the agent.
            terminated = self.win or (self.game_over and not self.win)
            truncated = self.steps >= self.MAX_STEPS and not terminated
            return self._get_observation(), 0, terminated, truncated, self._get_info()
            
        reward = -0.1  # Penalty for each step to encourage efficiency
        
        # --- Player Movement ---
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
        
        # Check boundaries before accessing the game map
        if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT and self.game_map[new_y, new_x] == 0:
            self.player_pos = (new_x, new_y)

        # --- Guard Movement ---
        for guard in self.guards:
            guard.move()

        # --- Check for Events ---
        # Artifact collection
        for i, artifact in enumerate(self.artifacts):
            if not self.artifacts_collected[i] and self.player_pos == artifact["pos"]:
                self.artifacts_collected[i] = True
                reward += 10
                self.score += 10

        # Guard detection
        for guard in self.guards:
            if self._is_player_detected(guard):
                self.game_over = True
                reward -= 50
                self.score -= 50
                break
        
        self.steps += 1
        
        # --- Check for Termination ---
        terminated = False
        truncated = False
        
        if all(self.artifacts_collected):
            self.win = True
            self.game_over = True
            terminated = True
            reward += 50
            self.score += 50
        
        if self.game_over and not self.win:
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        # Per Gymnasium API, terminated and truncated cannot both be true
        if terminated:
            truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

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
            "artifacts_collected": sum(self.artifacts_collected),
        }

    def _generate_map(self):
        self.game_map = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        stack = []
        
        start_x = self.np_random.integers(1, self.GRID_WIDTH - 1, endpoint=False)
        start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1, endpoint=False)
        if start_x % 2 == 0: start_x += 1
        if start_y % 2 == 0: start_y += 1
            
        self.game_map[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and self.game_map[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                self.game_map[ny, nx] = 0
                self.game_map[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_entities(self):
        floor_tiles = np.argwhere(self.game_map == 0)
        
        # Player (pos is x, y)
        player_idx = self.np_random.integers(len(floor_tiles))
        player_y, player_x = floor_tiles[player_idx]
        self.player_pos = (player_x, player_y)

        # Artifacts
        self.artifacts = []
        self.artifacts_collected = [False] * self.NUM_ARTIFACTS
        dead_ends = []
        for y, x in floor_tiles:
            neighbors = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self.game_map[y+dy, x+dx] == 0:
                    neighbors += 1
            if neighbors == 1:
                dead_ends.append((x, y))
        
        candidate_tiles = [tuple(t) for t in floor_tiles]
        candidate_tiles = [(x, y) for y, x in candidate_tiles] # convert to (x,y)
        
        if len(dead_ends) < self.NUM_ARTIFACTS:
            # Fallback if not enough dead ends
            candidate_positions = [pos for pos in candidate_tiles if pos != self.player_pos]
            artifact_indices = self.np_random.choice(len(candidate_positions), self.NUM_ARTIFACTS, replace=False)
            artifact_positions = [candidate_positions[i] for i in artifact_indices]
        else:
            candidate_positions = [pos for pos in dead_ends if pos != self.player_pos]
            if len(candidate_positions) < self.NUM_ARTIFACTS: # Handle player spawning on a dead end
                 candidate_positions = [pos for pos in candidate_tiles if pos != self.player_pos]
            artifact_indices = self.np_random.choice(len(candidate_positions), self.NUM_ARTIFACTS, replace=False)
            artifact_positions = [candidate_positions[i] for i in artifact_indices]

        for i in range(self.NUM_ARTIFACTS):
            self.artifacts.append({"pos": artifact_positions[i], "color": self.ARTIFACT_COLORS[i]})

        # Guards
        self.guards = []
        for _ in range(self.NUM_GUARDS):
            path = []
            while len(path) < 5: # Ensure a decent patrol length
                start_idx, end_idx = self.np_random.choice(len(floor_tiles), 2, replace=False)
                start_pos, end_pos = floor_tiles[start_idx], floor_tiles[end_idx]
                
                path_segment = []
                if self.np_random.random() > 0.5: # Horizontal
                    if start_pos[0] != end_pos[0]: continue
                    y = start_pos[0]
                    x1, x2 = sorted([start_pos[1], end_pos[1]])
                    path_segment = [(y, x) for x in range(x1, x2 + 1)]
                else: # Vertical
                    if start_pos[1] != end_pos[1]: continue
                    x = start_pos[1]
                    y1, y2 = sorted([start_pos[0], end_pos[0]])
                    path_segment = [(y, x) for y in range(y1, y2 + 1)]
                
                if path_segment and all(self.game_map[y, x] == 0 for y, x in path_segment):
                    path = [(x, y) for y, x in path_segment]
            
            self.guards.append(Guard(path, self.np_random))


    def _is_player_detected(self, guard):
        # 1. Bounding box and distance check
        px, py = self.player_pos
        gx, gy = guard.pos
        dist_sq = (px - gx)**2 + (py - gy)**2
        if dist_sq > guard.view_distance**2:
            return False

        # 2. Angle check
        player_vec = (px - gx, py - gy)
        angle_to_player = math.atan2(player_vec[1], player_vec[0])
        guard_angle = math.atan2(guard.facing_vector[1], guard.facing_vector[0])
        
        angle_diff = guard_angle - angle_to_player
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        if abs(angle_diff) > guard.view_angle / 2:
            return False

        # 3. Line of sight check (Bresenham's line algorithm)
        x1, y1 = gx, gy
        x2, y2 = px, py
        dx_line, dy_line = abs(x2 - x1), -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx_line + dy_line
        
        while True:
            if x1 == x2 and y1 == y2: break
            if (x1 != gx or y1 != gy) and (0 <= y1 < self.GRID_HEIGHT and 0 <= x1 < self.GRID_WIDTH and self.game_map[y1, x1] == 1):
                return False
            e2 = 2 * err
            if e2 >= dy_line:
                err += dy_line
                x1 += sx
            if e2 <= dx_line:
                err += dx_line
                y1 += sy
        
        return True

    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.game_map[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        for i, artifact in enumerate(self.artifacts):
            ax, ay = artifact["pos"]
            center_pixel = (int((ax + 0.5) * self.TILE_SIZE), int((ay + 0.5) * self.TILE_SIZE))
            pygame.gfxdraw.filled_circle(self.screen, center_pixel[0], center_pixel[1], self.TILE_SIZE // 2, self.COLOR_PEDESTAL)
            if not self.artifacts_collected[i]:
                radius = self.TILE_SIZE // 3
                pygame.gfxdraw.filled_circle(self.screen, center_pixel[0], center_pixel[1], radius, artifact["color"])
                pygame.gfxdraw.aacircle(self.screen, center_pixel[0], center_pixel[1], radius, artifact["color"])

        for guard in self.guards:
            if len(guard.path) > 1:
                path_pixels = [(int((x+0.5)*self.TILE_SIZE), int((y+0.5)*self.TILE_SIZE)) for x,y in guard.path]
                pygame.draw.lines(self.screen, self.COLOR_GUARD_PATH, False, path_pixels, 2)
            
            gx_p, gy_p = (guard.pos[0] + 0.5) * self.TILE_SIZE, (guard.pos[1] + 0.5) * self.TILE_SIZE
            p1 = (int(gx_p), int(gy_p))
            
            guard_angle = math.atan2(guard.facing_vector[1], guard.facing_vector[0])
            angle1 = guard_angle - guard.view_angle / 2
            angle2 = guard_angle + guard.view_angle / 2
            
            dist_pixels = guard.view_distance * self.TILE_SIZE
            p2 = (int(p1[0] + dist_pixels * math.cos(angle1)), int(p1[1] + dist_pixels * math.sin(angle1)))
            p3 = (int(p1[0] + dist_pixels * math.cos(angle2)), int(p1[1] + dist_pixels * math.sin(angle2)))
            
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_GUARD_VISION)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_GUARD_VISION)
            
            center_pixel = (int(gx_p), int(gy_p))
            radius = self.TILE_SIZE // 2 - 2
            pygame.gfxdraw.filled_circle(self.screen, center_pixel[0], center_pixel[1], radius, self.COLOR_GUARD)
            pygame.gfxdraw.aacircle(self.screen, center_pixel[0], center_pixel[1], radius, self.COLOR_GUARD)

        px, py = self.player_pos
        center_pixel = (int((px + 0.5) * self.TILE_SIZE), int((py + 0.5) * self.TILE_SIZE))
        
        glow_radius = int(self.TILE_SIZE * 0.8)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (center_pixel[0] - glow_radius, center_pixel[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        radius = self.TILE_SIZE // 2 - 2
        pygame.gfxdraw.filled_circle(self.screen, center_pixel[0], center_pixel[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_pixel[0], center_pixel[1], radius, self.COLOR_PLAYER)

    def _render_ui(self):
        artifact_text = f"Artifacts: {sum(self.artifacts_collected)} / {self.NUM_ARTIFACTS}"
        text_surface = self.font_ui.render(artifact_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        step_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        text_surface_steps = self.font_ui.render(step_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface_steps, (self.WIDTH - text_surface_steps.get_width() - 10, 10))

        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "DETECTED"
                color = self.COLOR_GUARD
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(bg_surf, bg_rect)
            
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


class Guard:
    def __init__(self, path, np_random):
        self.path = path
        self.np_random = np_random
        self.path_index = self.np_random.integers(0, len(path))
        self.pos = self.path[self.path_index]
        self.direction = self.np_random.choice([-1, 1])
        self.facing_vector = (1, 0)
        self.view_angle = math.pi / 2.5
        self.view_distance = 6.0

    def move(self):
        if not self.path or len(self.path) <= 1:
            return
            
        next_index = self.path_index + self.direction
        if not (0 <= next_index < len(self.path)):
            self.direction *= -1
            next_index = self.path_index + self.direction

        prev_pos = self.pos
        self.path_index = next_index
        self.pos = self.path[self.path_index]
        
        dx = self.pos[0] - prev_pos[0]
        dy = self.pos[1] - prev_pos[1]
        if dx != 0 or dy != 0:
            self.facing_vector = (dx, dy)


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    pygame.display.set_caption("Museum Heist")
    game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    done = False
    while running:
        action_key = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    action_key = key_map[event.key]
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if action_key != 0 and not done:
            obs, reward, terminated, truncated, info = env.step([action_key, 0, 0])
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()