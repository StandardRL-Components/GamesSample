import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Avoid the red vision cones of the guards."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down stealth game. Navigate the maze, collect yellow items, and reach the green exit without being spotted three times."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000
        self.NUM_ITEMS = 5
        self.NUM_GUARDS = 3
        self.DETECTION_LIMIT = 3

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (50, 60, 70)
        self.COLOR_FLOOR = (30, 35, 40) # Not drawn, but used for logic
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_EXIT_GLOW = (100, 255, 200)
        self.COLOR_ITEM = (255, 220, 0)
        self.COLOR_ITEM_GLOW = (255, 240, 100)
        self.COLOR_GUARD_FOV = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_DANGER = (255, 80, 80)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables
        self.player_pos = None
        self.exit_pos = None
        self.items = None
        self.guards = None
        self.walls = None
        
        self.steps = 0
        self.score = 0
        self.detections = 0
        self.items_collected = 0
        self.game_over = False
        self.win_message = ""
        
        self.np_random = None

    def _generate_level(self):
        """Generates a maze using randomized DFS, ensuring a path exists."""
        grid = [[True] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        walls = set()
        
        # Stack for DFS
        stack = deque()
        
        # FIX: Generate start_x and start_y within their respective grid bounds
        # The original code used GRID_WIDTH for both, causing an IndexError for start_y.
        start_x = self.np_random.integers(1, self.GRID_WIDTH)
        start_y = self.np_random.integers(1, self.GRID_HEIGHT)

        start_x |= 1; start_y |= 1 # Ensure odd starting point
        grid[start_y][start_x] = False
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and grid[ny][nx]:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # np_random.choice on a list of tuples requires converting it to a numpy array first
                chosen_idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[chosen_idx]
                grid[ny][nx] = False
                grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = False
                stack.append((nx, ny))
            else:
                stack.pop()
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if grid[y][x]:
                    walls.add((x, y))
        return walls

    def _get_open_tiles(self):
        return [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if (x, y) not in self.walls]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.walls = self._generate_level()
        open_tiles = self._get_open_tiles()
        
        if not open_tiles: # Handle the unlikely case of a fully filled maze
            # Fallback: create a minimal open space
            self.walls = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if x==0 or y==0 or x==self.GRID_WIDTH-1 or y==self.GRID_HEIGHT-1)
            open_tiles = self._get_open_tiles()

        self.np_random.shuffle(open_tiles)
        
        self.player_pos = open_tiles.pop()
        self.exit_pos = open_tiles.pop()
        
        self.items = set()
        for _ in range(min(self.NUM_ITEMS, len(open_tiles))):
            self.items.add(open_tiles.pop())

        self.guards = []
        for _ in range(min(self.NUM_GUARDS, len(open_tiles))):
            if not open_tiles: break
            start_pos = open_tiles.pop()
            path = [start_pos]
            # Create a simple back-and-forth patrol path
            for i in range(self.np_random.integers(2, 5)):
                if not open_tiles: break
                path.append(open_tiles.pop())
            self.guards.append(Guard(path, self.np_random))
            
        self.steps = 0
        self.score = 0
        self.detections = 0
        self.items_collected = 0
        self.game_over = False
        self.win_message = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01 # Small penalty per step to encourage speed

        # 1. Player Movement
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right

        if (px, py) not in self.walls and 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            self.player_pos = (px, py)

        # 2. Item Collection
        if self.player_pos in self.items:
            self.items.remove(self.player_pos)
            self.items_collected += 1
            reward += 10
            self.score += 10

        # 3. Guard Update and Detection
        is_detected_this_step = False
        for guard in self.guards:
            guard.update(self.walls)
            if self._is_player_in_fov(guard):
                is_detected_this_step = True
        
        if is_detected_this_step:
            self.detections += 1
            reward -= 20
            self.score -= 20

        # 4. Update Game State
        self.steps += 1
        
        # 5. Check Termination Conditions
        terminated = False
        truncated = False
        if self.player_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            reward += 100
            self.win_message = "SUCCESS"
        elif self.detections >= self.DETECTION_LIMIT:
            terminated = True
            self.game_over = True
            reward -= 50
            self.win_message = "SPOTTED"
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Or truncated = True
            self.game_over = True
            self.win_message = "TIME UP"
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _is_player_in_fov(self, guard):
        """Check distance, angle, and line of sight."""
        gx, gy = guard.pos
        px, py = self.player_pos
        
        dist = math.hypot(px - gx, py - gy)
        if dist > guard.fov_radius:
            return False

        angle_to_player = math.atan2(py - gy, px - gx)
        angle_diff = (angle_to_player - guard.angle + math.pi) % (2 * math.pi) - math.pi
        
        if abs(angle_diff) > guard.fov_angle / 2:
            return False
            
        # Line of sight check (Bresenham's line algorithm)
        x0, y0 = gx, gy
        x1, y1 = px, py
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            if (int(x0), int(y0)) in self.walls:
                return False
            if int(x0) == int(x1) and int(y0) == int(y1):
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        
        return True

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
            "detections": self.detections,
            "items_collected": self.items_collected,
        }

    def _render_game(self):
        ts = self.TILE_SIZE
        
        # Draw walls
        for x, y in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (x * ts, y * ts, ts, ts))
        
        # Draw exit
        ex, ey = self.exit_pos
        self._draw_glowing_rect(self.screen, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, (ex * ts, ey * ts, ts, ts), 5)
        
        # Draw items
        for ix, iy in self.items:
            self._draw_glowing_rect(self.screen, self.COLOR_ITEM, self.COLOR_ITEM_GLOW, (ix * ts + ts//4, iy * ts + ts//4, ts//2, ts//2), 3)

        # Draw guards
        for guard in self.guards:
            guard.draw(self.screen, ts, self.COLOR_GUARD_FOV)

        # Draw player
        px, py = self.player_pos
        center = (int(px * ts + ts / 2), int(py * ts + ts / 2))
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, center, ts // 3, 5)

    def _render_ui(self):
        # Detections
        detection_text = self.font_ui.render(f"DETECTIONS: {self.detections}/{self.DETECTION_LIMIT}", True, self.COLOR_UI_DANGER if self.detections > 0 else self.COLOR_UI_TEXT)
        self.screen.blit(detection_text, (10, 10))

        # Items
        item_text = self.font_ui.render(f"ITEMS: {self.items_collected}/{self.NUM_ITEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(item_text, (self.WIDTH - item_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.win_message, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius, glow_size):
        for i in range(glow_size, 0, -1):
            alpha = 150 * (1 - (i / glow_size))
            glow_surf = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, int(alpha)), (radius + i, radius + i), radius + i)
            surface.blit(glow_surf, (center[0] - radius - i, center[1] - radius - i))
        pygame.draw.circle(surface, color, center, radius)

    def _draw_glowing_rect(self, surface, color, glow_color, rect, glow_size):
        r = pygame.Rect(rect)
        for i in range(glow_size, 0, -1):
            alpha = 100 * (1 - (i / glow_size))
            glow_surf = pygame.Surface((r.width + i * 2, r.height + i * 2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*glow_color, int(alpha)), glow_surf.get_rect(), border_radius=3)
            surface.blit(glow_surf, (r.x - i, r.y - i))
        pygame.draw.rect(surface, color, r, border_radius=3)

    def close(self):
        pygame.quit()


class Guard:
    def __init__(self, path, np_random):
        self.path = path
        self.np_random = np_random
        self.path_index = 0
        self.pos = self.path[0]
        self.target = self.path[1] if len(self.path) > 1 else self.path[0]
        self.direction = (self.target[0] - self.pos[0], self.target[1] - self.pos[1])
        self.angle = math.atan2(self.direction[1], self.direction[0])
        self.fov_radius = 6 # in tiles
        self.fov_angle = math.pi / 2 # 90 degrees

    def update(self, walls):
        if not self.path or len(self.path) < 2:
            return

        if self.pos == self.target:
            self.path_index = (self.path_index + 1) % len(self.path)
            self.target = self.path[self.path_index]

        # Move towards target
        next_pos = self.pos
        dx, dy = self.target[0] - self.pos[0], self.target[1] - self.pos[1]
        
        if abs(dx) > abs(dy):
            next_pos = (self.pos[0] + np.sign(dx), self.pos[1])
        elif dy != 0:
            next_pos = (self.pos[0], self.pos[1] + np.sign(dy))
        
        if next_pos not in walls:
            self.pos = next_pos
            self.direction = (self.target[0] - self.pos[0], self.target[1] - self.pos[1])
            if self.direction != (0, 0):
                self.angle = math.atan2(self.direction[1], self.direction[0])

    def draw(self, surface, tile_size, color):
        # Draw guard body
        center_x = int(self.pos[0] * tile_size + tile_size / 2)
        center_y = int(self.pos[1] * tile_size + tile_size / 2)
        pygame.draw.circle(surface, (200, 0, 0), (center_x, center_y), tile_size // 3)

        # Draw FOV cone
        p1 = (center_x, center_y)
        
        angle1 = self.angle - self.fov_angle / 2
        p2_x = center_x + self.fov_radius * tile_size * math.cos(angle1)
        p2_y = center_y + self.fov_radius * tile_size * math.sin(angle1)
        
        angle2 = self.angle + self.fov_angle / 2
        p3_x = center_x + self.fov_radius * tile_size * math.cos(angle2)
        p3_y = center_y + self.fov_radius * tile_size * math.sin(angle2)
        
        points = [p1, (p2_x, p2_y), (p3_x, p3_y)]
        alpha_color = (*color, 100)
        pygame.gfxdraw.aapolygon(surface, points, alpha_color)
        pygame.gfxdraw.filled_polygon(surface, points, alpha_color)

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run main in a headless environment. Exiting.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()
        
        terminated = False
        
        print(env.user_guide)

        while not terminated:
            movement = 0 # No-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                pygame.time.wait(3000) # Pause for 3 seconds before closing
                
            clock.tick(15) # Control game speed for manual play
            
        env.close()