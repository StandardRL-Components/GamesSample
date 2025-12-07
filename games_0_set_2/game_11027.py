import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:30:38.534573
# Source Brief: brief_01027.md
# Brief Index: 1027
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Restore a polluted grid by planting bio-luminescent flora. Expand your safe haven while avoiding patrol drones."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to plant flora and shift at a hack point to stun a drone."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT
    
    VICTORY_PERCENTAGE = 0.75
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_POLLUTED = (60, 50, 55)
    COLOR_HAVEN = (20, 40, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_DRONE = (255, 50, 50)
    COLOR_DRONE_SCANNER = (255, 100, 100, 100)
    COLOR_PATH = (180, 160, 40, 80)
    COLOR_HACK = (255, 220, 0)
    COLOR_SAFE_PLANT = (0, 255, 200)
    COLOR_EXPOSED_PLANT = (255, 120, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Game parameters
    INITIAL_DRONES = 2
    DRONE_SPEED_START = 0.5
    DRONE_SPEED_INCREASE = 0.05
    DRONE_SPEED_MAX = 1.0
    DRONE_DETECTION_RADIUS = CELL_WIDTH * 1.5
    DRONE_STUN_DURATION = 15

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.grid = []
        self.drones = []
        self.particles = []
        self.cursor_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.total_restorable_cells = 0
        self.haven_cell_count = 0
        
        # self.reset() is called by the wrapper
        # self.validate_implementation() # No need to call this here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.particles = []
        
        self._initialize_grid()
        self._initialize_drones()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self._update_havens() # Initial haven calculation
        self.haven_cell_count = self._count_haven_cells()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Player Actions
        self._move_cursor(movement)
        
        if space_pressed:
            reward += self._action_plant()
        
        if shift_pressed:
            reward += self._action_hack()

        # 2. Update Game State
        self._update_drones()
        self._update_particles()
        self._update_difficulty()

        # 3. Calculate Rewards & Check Termination
        terminated = False
        truncated = False
        
        # Continuous penalty for exposed plants
        exposed_plants = self._count_exposed_plants()
        reward -= exposed_plants * 0.01

        # Check for drone detection
        if self._check_detection():
            self.game_over = True
            self.victory = False
            terminated = True
            reward = -100.0
        
        # Check for victory
        restoration_pct = self.haven_cell_count / self.total_restorable_cells if self.total_restorable_cells > 0 else 0
        if restoration_pct >= self.VICTORY_PERCENTAGE:
            self.game_over = True
            self.victory = True
            terminated = True
            reward = 100.0

        # Check for max steps
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Game Logic ---

    def _initialize_grid(self):
        self.grid = []
        light_map = self._generate_light_map()
        self.total_restorable_cells = 0

        for y in range(self.GRID_HEIGHT):
            row = []
            for x in range(self.GRID_WIDTH):
                cell = {
                    "type": "polluted",
                    "ambient_light": light_map[y][x],
                    "is_haven": False,
                    "plant_pulse_phase": self.np_random.uniform(0, 2 * math.pi)
                }
                row.append(cell)
                self.total_restorable_cells += 1
            self.grid.append(row)
        
        # Place rebel base and hack points
        self.grid[self.GRID_HEIGHT // 2][1]["type"] = "base"
        self.grid[self.GRID_HEIGHT // 2][1]["is_haven"] = True
        
        hack_points_placed = 0
        while hack_points_placed < 3:
            hx, hy = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if self.grid[hy][hx]["type"] == "polluted":
                self.grid[hy][hx]["type"] = "hack_point"
                self.total_restorable_cells -= 1
                hack_points_placed += 1

    def _generate_light_map(self):
        # Simple gradient light map for visual variety
        light_map = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH))
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # A mix of radial and linear gradients
                dist_from_center = math.sqrt((x - self.GRID_WIDTH/2)**2 + (y - self.GRID_HEIGHT/2)**2)
                max_dist = math.sqrt((self.GRID_WIDTH/2)**2 + (self.GRID_HEIGHT/2)**2)
                radial_light = 0.7 * (1 - dist_from_center / max_dist)
                linear_light = 0.3 * (x / self.GRID_WIDTH)
                light_map[y, x] = np.clip(radial_light + linear_light + self.np_random.uniform(-0.1, 0.1), 0.1, 1.0)
        return light_map

    def _initialize_drones(self):
        self.drones = []
        paths = [
            self._generate_drone_path((1, 1), (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)),
            self._generate_drone_path((self.GRID_WIDTH - 2, 1), (1, self.GRID_HEIGHT - 2), reverse=True)
        ]
        for i in range(self.INITIAL_DRONES):
            self._add_drone(paths[i])

    def _add_drone(self, path=None):
        if path is None:
            path = self._generate_drone_path()
        
        start_pos = self._grid_to_pixel(path[0])
        drone = {
            "pos": np.array(start_pos, dtype=float),
            "path": path,
            "path_index": 0,
            "speed": self.DRONE_SPEED_START,
            "stun_timer": 0
        }
        self.drones.append(drone)

    def _generate_drone_path(self, start=None, end=None, reverse=False):
        if start is None:
            start = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
        if end is None:
            end = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))

        # Simple rectangular path
        path = [(start[0], start[1]), (end[0], start[1]), (end[0], end[1]), (start[0], end[1])]
        if reverse:
            path.reverse()
        return path
    
    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        # Wrap around
        x %= self.GRID_WIDTH
        y %= self.GRID_HEIGHT
        self.cursor_pos = (x, y)

    def _action_plant(self):
        x, y = self.cursor_pos
        cell = self.grid[y][x]
        if cell["type"] == "polluted":
            cell["type"] = "plant"
            self._create_particles(self._grid_to_pixel((x, y)), 10, self.COLOR_SAFE_PLANT)
            
            prev_haven_count = self.haven_cell_count
            self._update_havens()
            self.haven_cell_count = self._count_haven_cells()
            new_haven_cells = self.haven_cell_count - prev_haven_count
            return new_haven_cells * 0.1
        return 0

    def _action_hack(self):
        x, y = self.cursor_pos
        if self.grid[y][x]["type"] == "hack_point":
            if self.drones:
                drone_to_stun = self.np_random.choice(self.drones)
                drone_to_stun["stun_timer"] = self.DRONE_STUN_DURATION
                self._create_particles(self._grid_to_pixel((x,y)), 20, self.COLOR_HACK)
                return 5.0
        return 0

    def _update_drones(self):
        for drone in self.drones:
            if drone["stun_timer"] > 0:
                drone["stun_timer"] -= 1
                continue

            target_node = drone["path"][drone["path_index"]]
            target_pos = np.array(self._grid_to_pixel(target_node), dtype=float)
            
            direction = target_pos - drone["pos"]
            distance = np.linalg.norm(direction)
            
            move_dist = drone["speed"] * self.CELL_WIDTH
            
            if distance < move_dist:
                drone["pos"] = target_pos
                drone["path_index"] = (drone["path_index"] + 1) % len(drone["path"])
            else:
                drone["pos"] += (direction / distance) * move_dist

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            for drone in self.drones:
                drone["speed"] = min(self.DRONE_SPEED_MAX, drone["speed"] + self.DRONE_SPEED_INCREASE)

        if self.steps > 0 and self.steps % 500 == 0:
            self._add_drone()

    def _update_havens(self):
        # Reset haven status
        for row in self.grid:
            for cell in row:
                cell["is_haven"] = False
        
        q = deque()
        # Find all base stations to start BFS from
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x]["type"] == "base":
                    q.append((x, y))
                    self.grid[y][x]["is_haven"] = True

        visited = set()

        while q:
            x, y = q.popleft()
            if (x, y) in visited:
                continue
            visited.add((x,y))
            
            self.grid[y][x]["is_haven"] = True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    neighbor_cell = self.grid[ny][nx]
                    if (neighbor_cell["type"] == "plant" or neighbor_cell["type"] == "base") and not neighbor_cell["is_haven"]:
                        q.append((nx, ny))

    def _check_detection(self):
        for drone in self.drones:
            if drone["stun_timer"] > 0:
                continue
            drone_pos = drone["pos"]
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    cell = self.grid[y][x]
                    if cell["type"] == "plant" and not cell["is_haven"]:
                        plant_pos = self._grid_to_pixel((x, y))
                        distance = np.linalg.norm(drone_pos - plant_pos)
                        if distance < self.DRONE_DETECTION_RADIUS:
                            return True
        return False

    def _count_haven_cells(self):
        count = 0
        for row in self.grid:
            for cell in row:
                if (cell["type"] == "plant" or cell["type"] == "base") and cell["is_haven"]:
                    count += 1
        return count

    def _count_exposed_plants(self):
        count = 0
        for row in self.grid:
            for cell in row:
                if cell["type"] == "plant" and not cell["is_haven"]:
                    count += 1
        return count

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid cells and ambient light
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell = self.grid[y][x]
                rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                
                bg_color = self.COLOR_HAVEN if cell["is_haven"] else self.COLOR_POLLUTED
                pygame.draw.rect(self.screen, bg_color, rect)

                # Ambient light effect
                light_radius = int(cell["ambient_light"] * self.CELL_WIDTH * 0.6)
                light_color = (40, 40, 60)
                self._draw_soft_circle(self.screen, light_color, rect.center, light_radius)

        # Draw drone paths
        for drone in self.drones:
            path_pixels = [self._grid_to_pixel(p) for p in drone["path"]]
            if len(path_pixels) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH, True, path_pixels, 2)

        # Draw grid elements (plants, etc.)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell = self.grid[y][x]
                pos = self._grid_to_pixel((x, y))
                
                if cell["type"] == "plant":
                    self._draw_plant(pos, cell)
                elif cell["type"] == "hack_point":
                    self._draw_hack_point(pos)
                elif cell["type"] == "base":
                    self._draw_base(pos)

        self._draw_particles()
        self._draw_drones()
        self._draw_cursor()

    def _render_ui(self):
        # Restoration percentage
        restoration_pct = 0
        if self.total_restorable_cells > 0:
            restoration_pct = self.haven_cell_count / self.total_restorable_cells
        
        restore_text = f"ECO-GRID: {restoration_pct:.1%}"
        text_surface = self.font_ui.render(restore_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Steps
        steps_text = f"CYCLE: {self.steps}/{self.MAX_STEPS}"
        text_surface = self.font_ui.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "ECOSYSTEM RESTORED" if self.victory else "MISSION FAILED"
            color = self.COLOR_SAFE_PLANT if self.victory else self.COLOR_DRONE
            text_surface = self.font_game_over.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _draw_plant(self, pos, cell):
        pulse = (math.sin(self.steps * 0.1 + cell["plant_pulse_phase"]) + 1) / 2
        color = self.COLOR_SAFE_PLANT if cell["is_haven"] else self.COLOR_EXPOSED_PLANT
        
        base_radius = int(self.CELL_WIDTH * 0.15)
        glow_radius = base_radius + int(pulse * 8)
        
        self._draw_soft_circle(self.screen, (*color, 60), pos, glow_radius)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], base_radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], base_radius, color)
        
    def _draw_hack_point(self, pos):
        size = int(self.CELL_WIDTH * 0.4)
        rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_HACK, rect, 2)
        pygame.gfxdraw.filled_trigon(self.screen, 
                                     rect.centerx, rect.top, 
                                     rect.left, rect.bottom, 
                                     rect.right, rect.bottom, 
                                     (*self.COLOR_HACK, 100))

    def _draw_base(self, pos):
        size = int(self.CELL_WIDTH * 0.5)
        rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_SAFE_PLANT, rect, 0, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), rect, 2, border_radius=4)

    def _draw_drones(self):
        for drone in self.drones:
            pos = (int(drone["pos"][0]), int(drone["pos"][1]))
            
            # Scanner radius
            if drone["stun_timer"] == 0:
                self._draw_soft_circle(self.screen, self.COLOR_DRONE_SCANNER, pos, int(self.DRONE_DETECTION_RADIUS))

            # Body
            size = int(self.CELL_WIDTH * 0.3)
            points = [
                (pos[0], pos[1] - size),
                (pos[0] + size, pos[1]),
                (pos[0], pos[1] + size),
                (pos[0] - size, pos[1]),
            ]
            color = self.COLOR_DRONE if drone["stun_timer"] == 0 else (100, 100, 150)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 150 + int(pulse * 105)
        color = (*self.COLOR_CURSOR, alpha)
        
        # Create a temporary surface for the glowing rectangle
        temp_surf = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), 2, border_radius=3)
        self.screen.blit(temp_surf, rect.topleft)

    # --- Particles ---

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, color, pos, 2)

    # --- Helpers & Info ---
    
    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return (int((x + 0.5) * self.CELL_WIDTH), int((y + 0.5) * self.CELL_HEIGHT))

    def _draw_soft_circle(self, surface, color, center, radius):
        """Draws a circle with a soft, fading edge."""
        if radius <= 0: return
        target_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        
        # Create a temporary surface with per-pixel alpha
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Draw the circle on the temporary surface
        pygame.draw.circle(surf, color, (radius, radius), radius)
        
        # Blit with a blend mode to the main screen
        surface.blit(surf, target_rect, special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "restoration": self.haven_cell_count / self.total_restorable_cells if self.total_restorable_cells > 0 else 0,
            "drones": len(self.drones)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block is for local testing and will not be executed by the evaluation server.
    # It will be run with the original SDL_VIDEODRIVER setting.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bioluminescent Grid")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Slow down for manual play
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)

    env.close()