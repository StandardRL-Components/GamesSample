import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:26:12.909566
# Source Brief: brief_02779.md
# Brief Index: 2779
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent builds underwater portal structures.

    The goal is to match falling tiles to create a continuous portal from a
    source at the bottom of the screen. Activating the portal teleports
    resources, which replenish the constantly depleting oxygen supply and
    increase the depth reached.

    The game ends when oxygen runs out, the target depth is reached, or a
    maximum number of steps is exceeded.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build underwater portal structures by matching falling tiles. "
        "Activate the portal to replenish oxygen and reach greater depths."
    )
    user_guide = (
        "Controls: ←→ to move the tile, ↑↓ to rotate. "
        "Press space to place the tile and shift to activate the portal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 15
    TILE_SIZE = 24
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH * TILE_SIZE) // 2
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT * TILE_SIZE) // 2 + 20

    # Colors (Bioluminescent Theme)
    COLOR_BG = (5, 10, 25)
    COLOR_GRID = (20, 40, 70)
    COLOR_TEXT = (220, 230, 255)
    COLOR_OXYGEN_HIGH = (0, 255, 150)
    COLOR_OXYGEN_MED = (255, 255, 0)
    COLOR_OXYGEN_LOW = (255, 50, 50)
    COLOR_RESOURCE = (255, 220, 50)
    COLOR_PORTAL_SOURCE = (150, 255, 255)

    # Tile definitions: {type: {'connections': np.array, 'color': tuple}}
    TILE_TYPES = {
        0: {"connections": np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]), "color": (50, 100, 255)}, # I
        1: {"connections": np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]), "color": (255, 100, 50)}, # L
        2: {"connections": np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]]), "color": (100, 255, 100)}, # T
        3: {"connections": np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]]), "color": (255, 50, 200)}, # U
    }

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)

        # Persistent state across resets
        self.target_depth = 100

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.oxygen = 100.0
        self.current_depth = 0
        self.resources = 0
        self.tile_fall_speed = 1.0

        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.portal_source_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)

        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.bubbles = [self._create_bubble() for _ in range(30)]
        self.active_portal_path = set()
        
        self._spawn_new_tile()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if not self.game_over:
            # Rotations
            if movement == 1: self._rotate_tile(clockwise=False) # Up -> Rotate Left
            if movement == 2: self._rotate_tile(clockwise=True)  # Down -> Rotate Right
            # Movements
            if movement == 3: self._move_tile(-1) # Left
            if movement == 4: self._move_tile(1)  # Right
            # Place Tile
            if space_pressed:
                # 'place tile' sound effect
                self._place_tile()
            # Activate Portal
            if shift_pressed:
                # 'activate portal' sound effect
                reward += self._activate_portal()

        # --- Game State Update ---
        self._update_world()
        
        # --- Reward Calculation ---
        reward -= 0.01  # Oxygen depletion penalty

        # --- Termination Check ---
        terminated = False
        if self.oxygen <= 0:
            terminated = True
            reward -= 100
            # 'game over' sound effect
            self.game_over = True
        elif self.current_depth >= self.target_depth:
            terminated = True
            reward += 100
            self.target_depth += 50 # Increase difficulty for next game
            # 'level complete' sound effect
            self.game_over = True
        elif self.steps >= 5000:
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic Helpers ---
    def _spawn_new_tile(self):
        max_tile_type = min(len(self.TILE_TYPES), 1 + self.steps // 1000)
        tile_type = self.np_random.integers(0, max_tile_type)
        self.falling_tile = {
            "type": tile_type,
            "rotation": self.np_random.integers(0, 4),
            "x": self.GRID_WIDTH // 2 - 1,
            "y_pixel": self.GRID_ORIGIN_Y - self.TILE_SIZE * 2,
        }
        if self._check_collision(self.falling_tile, self.falling_tile['x'], int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)):
             self.oxygen = 0 # Game over if spawn is blocked

    def _move_tile(self, dx):
        new_x = self.falling_tile['x'] + dx
        grid_y = int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
        if not self._check_collision(self.falling_tile, new_x, grid_y):
            self.falling_tile['x'] = new_x
            # 'move' sound effect

    def _rotate_tile(self, clockwise):
        self.falling_tile['rotation'] = (self.falling_tile['rotation'] + (1 if clockwise else -1)) % 4
        grid_y = int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
        if self._check_collision(self.falling_tile, self.falling_tile['x'], grid_y):
            # If rotation causes collision, revert it
            self.falling_tile['rotation'] = (self.falling_tile['rotation'] + (-1 if clockwise else 1)) % 4
        else:
            # 'rotate' sound effect
            pass

    def _place_tile(self):
        grid_y = int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
        
        # Drop the tile to the lowest possible valid position
        while not self._check_collision(self.falling_tile, self.falling_tile['x'], grid_y + 1):
            grid_y += 1
        
        self.falling_tile['y_pixel'] = self.GRID_ORIGIN_Y + grid_y * self.TILE_SIZE
        self._lock_tile()

    def _lock_tile(self):
        shape = self._get_rotated_shape(self.falling_tile['type'], self.falling_tile['rotation'])
        grid_y = int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
        
        placed_any = False
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell == 1:
                    gx, gy = self.falling_tile['x'] + c, grid_y + r
                    if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
                        self.grid[gy][gx] = (self.falling_tile['type'], self.falling_tile['rotation'])
                        placed_any = True

        if placed_any:
            # 'tile lock' sound effect
            self._spawn_new_tile()
        else: # Tile placed completely out of bounds
            self._spawn_new_tile()

    def _check_collision(self, tile, grid_x, grid_y):
        shape = self._get_rotated_shape(tile['type'], tile['rotation'])
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell == 1:
                    gx, gy = grid_x + c, grid_y + r
                    if not (0 <= gx < self.GRID_WIDTH): return True  # Collision with side walls
                    if gy >= self.GRID_HEIGHT: return True # Collision with floor
                    if gy >= 0 and self.grid[gy][gx] is not None: return True # Collision with other tiles
        return False

    def _activate_portal(self):
        reward = 0
        self.active_portal_path.clear()
        
        q = deque()
        visited = set()
        
        source_x, source_y = self.portal_source_pos
        if source_y < self.GRID_HEIGHT and source_x < self.GRID_WIDTH and self.grid[source_y][source_x] is not None:
            q.append((source_x, source_y))
            visited.add((source_x, source_y))

        while q:
            x, y = q.popleft()
            self.active_portal_path.add((x,y))

            current_connections = self._get_tile_connections(x, y)
            
            # Check neighbors
            for dx, dy, from_dir, to_dir in [ (0, -1, "up", "down"), (0, 1, "down", "up"), (-1, 0, "left", "right"), (1, 0, "right", "left")]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[ny][nx] is not None:
                        neighbor_connections = self._get_tile_connections(nx, ny)
                        if current_connections[from_dir] and neighbor_connections[to_dir]:
                            visited.add((nx, ny))
                            q.append((nx, ny))
        
        if visited:
            # 'resource collect' sound effect
            portal_power = len(visited)
            self.resources += portal_power
            self.oxygen = min(100.0, self.oxygen + portal_power * 1.5)
            self.current_depth += portal_power
            reward += 0.1 * portal_power + 1.0 # Per-resource reward + completion reward

            # Spawn particles from activated tiles
            for (px, py) in visited:
                for _ in range(3):
                    self.particles.append(self._create_particle(px, py))
        return reward
    
    def _update_world(self):
        # Update falling tile
        if not self.game_over:
            self.falling_tile['y_pixel'] += self.tile_fall_speed
            grid_y = int((self.falling_tile['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
            if self._check_collision(self.falling_tile, self.falling_tile['x'], grid_y):
                self.falling_tile['y_pixel'] -= self.tile_fall_speed # Revert move
                self._lock_tile()
        
        # Update oxygen
        self.oxygen = max(0, self.oxygen - 0.02)

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.tile_fall_speed = min(5.0, self.tile_fall_speed + 0.05)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)
        
        # Update bubbles
        for b in self.bubbles:
            b['pos'][1] -= b['speed']
            b['pos'][0] += math.sin(b['pos'][1] / b['freq']) * b['amp']
            if b['pos'][1] < -b['radius']:
                self.bubbles.remove(b)
                self.bubbles.append(self._create_bubble())


    # --- State & Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_background_effects()
        self._draw_grid()
        self._draw_placed_tiles()
        if not self.game_over:
            self._draw_falling_tile_ghost()
            self._draw_tile(self.screen, self.falling_tile['type'], self.falling_tile['rotation'], self.falling_tile['x'], self.falling_tile['y_pixel'], is_falling=True)
        self._draw_particles()

    def _render_ui(self):
        # Oxygen Bar
        bar_x, bar_y, bar_w, bar_h = 20, 20, 200, 20
        fill_w = max(0, int(bar_w * (self.oxygen / 100.0)))
        
        if self.oxygen > 60: color = self.COLOR_OXYGEN_HIGH
        elif self.oxygen > 30: color = self.COLOR_OXYGEN_MED
        else: color = self.COLOR_OXYGEN_LOW
        
        pygame.draw.rect(self.screen, (30,30,30), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)
        
        # Text Info
        depth_text = self.font_ui.render(f"DEPTH: {self.current_depth}m / {self.target_depth}m", True, self.COLOR_TEXT)
        self.screen.blit(depth_text, (self.SCREEN_WIDTH - 220, 20))
        
        resource_text = self.font_ui.render(f"RESOURCES: {self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - 220, 45))
        
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "depth": self.current_depth, "oxygen": self.oxygen}

    # --- Drawing Helpers ---
    def _draw_background_effects(self):
        for b in self.bubbles:
            pygame.gfxdraw.filled_circle(self.screen, int(b['pos'][0]), int(b['pos'][1]), int(b['radius']), b['color'])
            pygame.gfxdraw.aacircle(self.screen, int(b['pos'][0]), int(b['pos'][1]), int(b['radius']), b['color'])

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_ORIGIN_X + x * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_ORIGIN_Y), (px, self.GRID_ORIGIN_Y + self.GRID_HEIGHT * self.TILE_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_ORIGIN_Y + y * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_ORIGIN_X, py), (self.GRID_ORIGIN_X + self.GRID_WIDTH * self.TILE_SIZE, py))

        # Draw portal source
        sx, sy = self.portal_source_pos
        rect = pygame.Rect(self.GRID_ORIGIN_X + sx * self.TILE_SIZE, self.GRID_ORIGIN_Y + sy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        self._draw_glow_rect(self.screen, self.COLOR_PORTAL_SOURCE, rect, 10)

    def _draw_placed_tiles(self):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell is not None:
                    is_active = (x, y) in self.active_portal_path
                    self._draw_tile(self.screen, cell[0], cell[1], x, self.GRID_ORIGIN_Y + y * self.TILE_SIZE, is_active=is_active)

    def _draw_falling_tile_ghost(self):
        ghost = self.falling_tile.copy()
        grid_y = int((ghost['y_pixel'] - self.GRID_ORIGIN_Y) / self.TILE_SIZE)
        while not self._check_collision(ghost, ghost['x'], grid_y + 1):
            grid_y += 1
        
        self._draw_tile(self.screen, ghost['type'], ghost['rotation'], ghost['x'], self.GRID_ORIGIN_Y + grid_y * self.TILE_SIZE, is_ghost=True)

    def _draw_tile(self, surface, tile_type, rotation, grid_x, y_pixel, is_falling=False, is_ghost=False, is_active=False):
        shape = self._get_rotated_shape(tile_type, rotation)
        color = self.TILE_TYPES[tile_type]['color']
        
        if is_ghost:
            color = (*color, 50) # Transparent
        elif is_falling:
            color = tuple(min(255, c + 50) for c in color)
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell == 1:
                    px = self.GRID_ORIGIN_X + (grid_x + c) * self.TILE_SIZE
                    
                    # Correct y-pixel calculation for placed tiles
                    if is_falling or is_ghost:
                        py = y_pixel + r * self.TILE_SIZE
                    else:
                        py = self.GRID_ORIGIN_Y + (y_pixel - self.GRID_ORIGIN_Y) + r * self.TILE_SIZE

                    rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
                    
                    if is_ghost:
                        pygame.draw.rect(surface, color, rect, 2)
                    else:
                        inner_rect = rect.inflate(-4, -4)
                        pygame.draw.rect(surface, color, inner_rect, 0, 4)
                        pygame.draw.rect(surface, tuple(min(255, c+60) for c in color), inner_rect, 1, 4)
                        if is_active:
                            self._draw_glow_rect(surface, self.COLOR_RESOURCE, rect, 5)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 50.0))))
            color = (*self.COLOR_RESOURCE, alpha)
            self._draw_glow_circle(self.screen, color, p['pos'], p['size'], 3)
    
    def _draw_glow_rect(self, surface, color, rect, radius):
        for i in range(radius, 0, -1):
            alpha = int(150 * (1 - i / radius))
            glow_color = (*color, alpha)
            glow_rect = rect.inflate(i*2, i*2)
            pygame.draw.rect(surface, glow_color, glow_rect, 1, 8)

    def _draw_glow_circle(self, surface, color, center, radius, width):
        for i in range(width, 0, -1):
            alpha = color[3] if len(color) == 4 else 255
            glow_alpha = int(alpha * (1 - i / width) * 0.5)
            glow_color = (*color[:3], glow_alpha)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius + i, glow_color)
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius + i, glow_color)

    # --- Utility Methods ---
    def _get_rotated_shape(self, tile_type, rotation):
        return np.rot90(self.TILE_TYPES[tile_type]['connections'], k=-rotation)

    def _get_tile_connections(self, x, y):
        tile_data = self.grid[y][x]
        if tile_data is None:
            return {"up": 0, "down": 0, "left": 0, "right": 0}
        
        shape = self._get_rotated_shape(tile_data[0], tile_data[1])
        return {
            "up": shape[0][1],
            "down": shape[2][1],
            "left": shape[1][0],
            "right": shape[1][2]
        }
    
    def _create_particle(self, grid_x, grid_y):
        return {
            'pos': [self.GRID_ORIGIN_X + (grid_x + 0.5) * self.TILE_SIZE, self.GRID_ORIGIN_Y + (grid_y + 0.5) * self.TILE_SIZE],
            'vel': [(self.np_random.random() - 0.5) * 1, (self.np_random.random() - 0.5) * 1 - 1],
            'life': self.np_random.integers(30, 60),
            'size': self.np_random.integers(2, 4)
        }

    def _create_bubble(self):
        return {
            'pos': [self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT + self.SCREEN_HEIGHT],
            'radius': self.np_random.random() * 3 + 1,
            'speed': self.np_random.random() * 1.5 + 0.5,
            'color': (100, 150, 255, self.np_random.integers(20, 80)),
            'amp': self.np_random.random() * 2,
            'freq': self.np_random.random() * 40 + 20
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Pygame setup for rendering
    pygame.display.set_caption("Deep Sea Constructor")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0
                elif event.key == pygame.K_SPACE: action[1] = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}, Depth: {info['depth']}m")
    env.close()