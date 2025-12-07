
# Generated: 2025-08-27T23:37:44.681438
# Source Brief: brief_03518.md
# Brief Index: 3518

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    user_guide = "Controls: ←→ to move the falling tile. Press Space to drop it quickly."

    # Must be a short, user-facing description of the game:
    game_description = "Build the tallest, most stable tower you can by placing falling tiles. Reach a height of 20 to win, but be careful - unstable placements will cause a collapse! You have 60 seconds."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Game constants
    WIN_HEIGHT = 20
    GRID_WIDTH = 11  # Horizontal grid size
    TILE_FALL_SPEED = 2.0
    TILE_MOVE_SPEED = 0.25 # grid units per step
    BASE_PLATFORM_WIDTH = 5

    # Visuals
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_UI_TEXT = (220, 220, 230)
    TILE_COLORS = [
        (52, 152, 219),  # Blue
        (231, 76, 60),   # Red
        (46, 204, 113),  # Green
        (241, 196, 15),  # Yellow
        (155, 89, 182),  # Purple
    ]
    
    # Isometric drawing parameters
    TILE_VISUAL_WIDTH = 48
    TILE_VISUAL_HEIGHT = 24
    TILE_VISUAL_DEPTH = 20 # Height of the cube side

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Etc...        
        self.np_random = None
        
        # Initialize state variables
        self.placed_tiles = None
        self.falling_tile = None
        self.particles = None
        self.steps = None
        self.score = None
        self.height = None
        self.game_over = None
        self.wobble_timer = None
        self.wobble_amplitude = None

        self.reset()
        
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.height = 0
        self.game_over = False
        self.wobble_timer = 0
        self.wobble_amplitude = 0.0

        self.placed_tiles = {}  # Using a dict for {(x, z): color}
        self.particles = []

        # Create the base platform
        base_y = 0
        base_x_start = (self.GRID_WIDTH - self.BASE_PLATFORM_WIDTH) // 2
        for i in range(self.BASE_PLATFORM_WIDTH):
            self.placed_tiles[(base_x_start + i, base_y)] = (100, 110, 120)

        self._spawn_tile()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = 0

        # Action Handling
        if movement == 3:  # Left
            self.falling_tile['x'] -= self.TILE_MOVE_SPEED
        elif movement == 4: # Right
            self.falling_tile['x'] += self.TILE_MOVE_SPEED

        self.falling_tile['x'] = np.clip(self.falling_tile['x'], 0, self.GRID_WIDTH - 1)

        # Game Logic
        if space_held:
            self.falling_tile['y'] += self.TILE_FALL_SPEED * 5
        else:
            self.falling_tile['y'] += self.TILE_FALL_SPEED
            
        if self.wobble_timer > 0:
            self.wobble_timer -= 1
        else:
            self.wobble_amplitude = 0.0

        # Check for landing
        land_z = self._get_support_z(self.falling_tile['x'])
        land_screen_y = self._grid_to_screen(0, land_z)[1] - self.TILE_VISUAL_HEIGHT / 2

        if self.falling_tile['y'] >= land_screen_y:
            new_tile_x = int(round(self.falling_tile['x']))
            new_tile_z = land_z + 1
            
            is_stable, is_risky = self._check_stability(new_tile_x, new_tile_z)

            if is_stable:
                self.placed_tiles[(new_tile_x, new_tile_z)] = self.falling_tile['color']
                self._create_particles(new_tile_x, new_tile_z, self.falling_tile['color'], 10)
                # sfx: tile_place.wav
                
                reward += 0.1
                if is_risky:
                    reward -= 0.02
                    self.wobble_timer = 30
                    self.wobble_amplitude = 2.0

                if new_tile_z > self.height:
                    self.height = new_tile_z
                    self.score += self.height
                    reward += 1.0

                if self.height >= self.WIN_HEIGHT:
                    self.game_over = True
                    reward += 100
                else:
                    self._spawn_tile()
            else:
                self.game_over = True
                reward -= 100
                self._trigger_collapse(new_tile_x, new_tile_z)
                # sfx: tower_collapse.wav
        
        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 10 # Timeout penalty

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_tower()
        if not self.game_over:
            self._render_falling_tile()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.height,
            "game_over": self.game_over,
        }

    # --- Helper Methods ---
    def _spawn_tile(self):
        self.falling_tile = {
            'x': self.GRID_WIDTH / 2,
            'y': 50,
            'color': self.np_random.choice(self.TILE_COLORS).tolist()
        }

    def _get_support_z(self, x_pos):
        grid_x = int(round(x_pos))
        max_z_in_col = -1
        for (gx, gz) in self.placed_tiles.keys():
            if gx == grid_x:
                max_z_in_col = max(max_z_in_col, gz)
        return max_z_in_col

    def _check_stability(self, x, z):
        if z == 0:
            return True, False
        
        support_z = z - 1
        
        direct_support = (x, support_z) in self.placed_tiles
        bridge_support_left = (x - 1, support_z) in self.placed_tiles
        bridge_support_right = (x + 1, support_z) in self.placed_tiles
        cantilever_left = (x - 1, support_z) in self.placed_tiles and not direct_support
        cantilever_right = (x + 1, support_z) in self.placed_tiles and not direct_support

        if direct_support: return True, False
        if bridge_support_left and bridge_support_right: return True, False
        if cantilever_left or cantilever_right: return True, True
            
        return False, False

    def _trigger_collapse(self, fail_x, fail_z):
        self._create_particles(fail_x, fail_z, self.falling_tile['color'], 50, is_collapse=True)
        tiles_to_remove = [k for k in self.placed_tiles if k[1] >= fail_z - 1]
        
        for tile in tiles_to_remove:
            if tile in self.placed_tiles:
                color = self.placed_tiles[tile]
                self._create_particles(tile[0], tile[1], color, 20, is_collapse=True)
                del self.placed_tiles[tile]

    def _create_particles(self, grid_x, grid_z, color, count, is_collapse=False):
        sx, sy = self._grid_to_screen(grid_x, grid_z)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi) if is_collapse else self.np_random.uniform(-math.pi, 0)
            speed = self.np_random.uniform(2, 6) if is_collapse else self.np_random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append([sx, sy, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.2
            p[4] -= 1

    # --- Rendering Methods ---
    def _grid_to_screen(self, grid_x, grid_z):
        wobble_x = math.sin(self.steps * 0.5) * self.wobble_amplitude * (self.wobble_timer / 30.0)
        screen_x = (self.SCREEN_WIDTH / 2) + (grid_x - self.GRID_WIDTH / 2) * (self.TILE_VISUAL_WIDTH / 2) + wobble_x
        screen_y = self.SCREEN_HEIGHT - 60 - (grid_z * self.TILE_VISUAL_DEPTH)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, x, y, color, shadow=True):
        color_light = tuple(min(255, c + 30) for c in color)
        color_dark = tuple(max(0, c - 30) for c in color)
        w, h, d = self.TILE_VISUAL_WIDTH, self.TILE_VISUAL_HEIGHT, self.TILE_VISUAL_DEPTH
        
        top_points = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
        left_points = [(x - w / 2, y), (x, y + h / 2), (x, y + h / 2 + d), (x - w / 2, y + d)]
        right_points = [(x + w / 2, y), (x, y + h / 2), (x, y + h / 2 + d), (x + w / 2, y + d)]
        
        if shadow:
            shadow_points = [(p[0] + 4, p[1] + 2) for p in right_points] + [(left_points[3][0] + 4, left_points[3][1] + 2), (left_points[2][0] + 4, left_points[2][1] + 2)]
            pygame.gfxdraw.filled_polygon(surface, shadow_points, (0, 0, 0, 40))

        pygame.gfxdraw.filled_polygon(surface, right_points, color_dark)
        pygame.gfxdraw.aapolygon(surface, right_points, color_dark)
        pygame.gfxdraw.filled_polygon(surface, left_points, color)
        pygame.gfxdraw.aapolygon(surface, left_points, color)
        pygame.gfxdraw.filled_polygon(surface, top_points, color_light)
        pygame.gfxdraw.aapolygon(surface, top_points, color_light)

    def _render_background(self):
        for i in range(1, self.WIN_HEIGHT + 1):
            if i % 5 == 0:
                y = self._grid_to_screen(0, i)[1] + self.TILE_VISUAL_HEIGHT / 2 + self.TILE_VISUAL_DEPTH / 2
                start_x = self.SCREEN_WIDTH / 2 - (self.GRID_WIDTH / 2) * (self.TILE_VISUAL_WIDTH / 2) - 30
                end_x = self.SCREEN_WIDTH / 2 + (self.GRID_WIDTH / 2) * (self.TILE_VISUAL_WIDTH / 2) + 30
                pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, y), (end_x, y), 1)
                text = self.font_small.render(str(i), True, self.COLOR_UI_TEXT)
                self.screen.blit(text, (start_x - 25, y - 8))

    def _render_tower(self):
        sorted_tiles = sorted(self.placed_tiles.items(), key=lambda item: (item[0][1], item[0][0]))
        for (gx, gz), color in sorted_tiles:
            sx, sy = self._grid_to_screen(gx, gz)
            depth_factor = max(0.5, 1.0 - gz * 0.02)
            darkened_color = tuple(int(c * depth_factor) for c in color)
            self._draw_iso_tile(self.screen, sx, sy, darkened_color)

    def _render_falling_tile(self):
        support_z = self._get_support_z(self.falling_tile['x'])
        shadow_x, shadow_y = self._grid_to_screen(self.falling_tile['x'], support_z + 1)
        self._draw_iso_tile(self.screen, shadow_x, shadow_y, (0,0,0,100), shadow=False)
        
        sx = (self.SCREEN_WIDTH / 2) + (self.falling_tile['x'] - self.GRID_WIDTH / 2) * (self.TILE_VISUAL_WIDTH / 2)
        sy = self.falling_tile['y']
        self._draw_iso_tile(self.screen, int(sx), int(sy), self.falling_tile['color'])

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), max(1, int(p[4] / 8)))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        height_text = self.font_main.render(f"HEIGHT: {self.height}/{self.WIN_HEIGHT}", True, self.COLOR_UI_TEXT)
        text_rect = height_text.get_rect(centerx=self.SCREEN_WIDTH / 2, top=10)
        self.screen.blit(height_text, text_rect)
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_main.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(right=self.SCREEN_WIDTH - 20, top=10)
        self.screen.blit(time_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "TOWER COMPLETE!" if self.height >= self.WIN_HEIGHT else "TOWER COLLAPSED!"
            end_text_color = (100, 255, 150) if self.height >= self.WIN_HEIGHT else (255, 100, 100)
            end_text = self.font_main.render(end_text_str, True, end_text_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        pygame.display.init()
    except pygame.error:
        try:
            os.environ['SDL_VIDEODRIVER'] = 'windows'
            pygame.display.init()
        except pygame.error:
             os.environ['SDL_VIDEODRIVER'] = 'dummy'


    env = GameEnv(render_mode="rgb_array")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Builder")
    obs, info = env.reset()
    done = False
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        env.clock.tick(env.FPS)
        
    print(f"Game Over. Final Score: {info['score']}, Final Height: {info['height']}")
    env.close()