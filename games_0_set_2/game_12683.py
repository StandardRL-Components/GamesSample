import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Quantum Tiles: A puzzle game where the player places quantum tiles to trigger
    chain reactions, aiming to match a target particle configuration.
    
    The environment follows the Gymnasium API and prioritizes high-quality visuals
    and engaging gameplay feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = "Place quantum tiles to trigger chain reactions and replicate a target particle pattern."
    user_guide = "Use arrow keys (↑↓←→) to move the cursor. Press space to place a tile and shift to cycle tile types."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 8
    CELL_SIZE = 40
    MAX_EPISODE_STEPS = 1000

    # Game element identifiers
    EMPTY = 0
    PARTICLE = 1
    TILE_A = 2  # Magenta
    TILE_B = 3  # Cyan
    TILE_C = 4  # Yellow
    
    # Visuals
    COLOR_BG = (16, 16, 40) # #101028
    COLOR_GRID = (32, 32, 64) # #202040
    COLOR_CURSOR = (255, 255, 0)
    COLOR_PARTICLE = (255, 255, 255)
    TILE_COLORS = {
        TILE_A: (255, 0, 255),
        TILE_B: (0, 255, 255),
        TILE_C: (255, 255, 0),
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_title = pygame.font.SysFont("monospace", 14, bold=True)

        # --- Game State (non-resettable) ---
        self.consecutive_wins = 0

        # --- Game State (resettable) ---
        # These are initialized in reset() to ensure clean episodes
        self.grid = None
        self.target_config = None
        self.cursor_pos = None
        self.render_cursor_pos = None
        self.moves_left = None
        self.max_moves = None
        self.available_tiles = [self.TILE_A, self.TILE_B, self.TILE_C]
        self.active_tile_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.effects_particles = []

        # --- Grid positioning ---
        self.grid_pixel_width = self.GRID_COLS * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_ROWS * self.CELL_SIZE
        self.grid_origin_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.render_cursor_pos = [
            self.grid_origin_x + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.grid_origin_y + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        ]
        
        self.active_tile_idx = 0
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True # Prevent action on first frame
        self.effects_particles = []

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions (on rising edge) ---
        # 1. Cycle active tile
        if shift_held and not self.prev_shift_held:
            self.active_tile_idx = (self.active_tile_idx + 1) % len(self.available_tiles)

        # 2. Move cursor
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_ROWS - 1)

        # 3. Place tile
        if space_held and not self.prev_space_held:
            place_x, place_y = self.cursor_pos
            
            if self.grid[place_y][place_x] == self.EMPTY:
                self.moves_left -= 1
                
                active_tile = self.available_tiles[self.active_tile_idx]
                self.grid[place_y][place_x] = active_tile
                
                chain_length = self._handle_collapse(place_x, place_y)
                if chain_length >= 3:
                    reward += 5.0
                elif chain_length > 0:
                    pass
            else:
                reward -= 0.1 # Penalty for invalid placement

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Continuous Reward & State Update ---
        current_match_count = self._count_matching_particles()
        reward += current_match_count
        self.score += reward

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self._check_win_condition():
            reward += 100.0
            self.score += 100.0
            self.consecutive_wins += 1
            terminated = True
        elif self.moves_left <= 0:
            reward -= 100.0
            self.score -= 100.0
            self.consecutive_wins = 0
            terminated = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            self.consecutive_wins = 0
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated

        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        # Difficulty scales every 5 wins
        difficulty_tier = self.consecutive_wins // 5
        num_target_particles = 2 + difficulty_tier
        self.max_moves = 50 + 5 * difficulty_tier
        self.moves_left = self.max_moves
        
        self.grid = [[self.EMPTY for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        # Loop until a valid, non-empty target is created
        while True:
            # Generate a solvable target configuration
            temp_grid = [[self.EMPTY for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
            self.target_config = [[self.EMPTY for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
            
            # Place random tiles to create a basis for the target
            for _ in range(num_target_particles * 2): # Place more tiles than needed
                x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
                tile = self.np_random.choice(self.available_tiles)
                if temp_grid[y][x] == self.EMPTY:
                    temp_grid[y][x] = tile

            # Simulate collapses to form the target
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if temp_grid[r][c] > self.PARTICLE:
                        component = self._find_connected_component(c, r, temp_grid)
                        if len(component) >= 2:
                            for c_x, c_y in component:
                                self.target_config[c_y][c_x] = self.PARTICLE
                                temp_grid[c_y][c_x] = self.EMPTY # Prevent re-processing
            
            # Check if the target is non-empty
            if any(self.PARTICLE in row for row in self.target_config):
                break # Valid target found, exit loop

    def _handle_collapse(self, x, y):
        component = self._find_connected_component(x, y, self.grid)
        if len(component) >= 2:
            for c_x, c_y in component:
                self.grid[c_y][c_x] = self.PARTICLE
                # Spawn visual effect particles
                px, py = self._grid_to_pixel(c_x, c_y)
                for _ in range(10):
                    self.effects_particles.append(EffectParticle(px, py, self.COLOR_PARTICLE))
            return len(component)
        return 0

    def _find_connected_component(self, start_x, start_y, grid):
        tile_type = grid[start_y][start_x]
        if tile_type <= self.PARTICLE:
            return []
            
        component = set()
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            component.add((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                    if grid[ny][nx] == tile_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return list(component)

    def _check_win_condition(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                is_particle = self.grid[r][c] == self.PARTICLE
                is_target = self.target_config[r][c] == self.PARTICLE
                if is_particle != is_target:
                    return False
        return True

    def _count_matching_particles(self):
        count = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == self.PARTICLE and self.target_config[r][c] == self.PARTICLE:
                    count += 1
        return count

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "consecutive_wins": self.consecutive_wins
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _grid_to_pixel(self, x, y):
        return (
            self.grid_origin_x + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.grid_origin_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_background()
        self._render_grid_and_particles()
        self._render_cursor()
        self._render_target_preview()
        self._update_and_render_effects()

    def _render_background(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.grid_origin_y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_origin_x, y), (self.grid_origin_x + self.grid_pixel_width, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.grid_origin_x + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_origin_y), (x, self.grid_origin_y + self.grid_pixel_height), 1)

    def _render_grid_and_particles(self):
        tile_size = int(self.CELL_SIZE * 0.8)
        particle_radius = int(self.CELL_SIZE * 0.35)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                px, py = self._grid_to_pixel(c, r)
                cell_state = self.grid[r][c]
                if cell_state in self.TILE_COLORS:
                    color = self.TILE_COLORS[cell_state]
                    rect = pygame.Rect(px - tile_size // 2, py - tile_size // 2, tile_size, tile_size)
                    self._draw_glowing_rect(self.screen, color, rect, 10)
                elif cell_state == self.PARTICLE:
                    self._draw_glowing_circle(self.screen, self.COLOR_PARTICLE, (px, py), particle_radius, 15)

    def _render_cursor(self):
        target_x, target_y = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        # Interpolate for smooth movement
        self.render_cursor_pos[0] += (target_x - self.render_cursor_pos[0]) * 0.5
        self.render_cursor_pos[1] += (target_y - self.render_cursor_pos[1]) * 0.5
        
        x, y = int(self.render_cursor_pos[0]), int(self.render_cursor_pos[1])
        size = int(self.CELL_SIZE * 0.5)
        points = [
            (x - size, y - size), (x + size, y - size),
            (x + size, y + size), (x - size, y + size)
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def _render_target_preview(self):
        preview_cell_size = 10
        preview_w = self.GRID_COLS * preview_cell_size
        preview_h = self.GRID_ROWS * preview_cell_size
        preview_x = self.SCREEN_WIDTH - preview_w - 20
        preview_y = self.SCREEN_HEIGHT - preview_h - 20
        
        title_surf = self.font_title.render("TARGET", True, (150, 150, 180))
        self.screen.blit(title_surf, (preview_x, preview_y - 20))

        pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x-2, preview_y-2, preview_w+4, preview_h+4), 1)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.target_config[r][c] == self.PARTICLE:
                    px = preview_x + c * preview_cell_size + preview_cell_size // 2
                    py = preview_y + r * preview_cell_size + preview_cell_size // 2
                    pygame.gfxdraw.filled_circle(self.screen, px, py, 3, (200, 200, 255))

    def _update_and_render_effects(self):
        for p in self.effects_particles[:]:
            p.update()
            if p.life <= 0:
                self.effects_particles.remove(p)
            else:
                p.draw(self.screen)

    def _render_ui(self):
        # Moves Left
        moves_text = f"MOVES: {self.moves_left}/{self.max_moves}"
        moves_surf = self.font_ui.render(moves_text, True, (200, 200, 255))
        self.screen.blit(moves_surf, (20, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, (200, 200, 255))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 10))
        
        # Current Tile Preview
        preview_x, preview_y = 60, self.SCREEN_HEIGHT - 50
        title_surf = self.font_title.render("ACTIVE TILE", True, (150, 150, 180))
        self.screen.blit(title_surf, (preview_x - 38, preview_y - 45))
        
        active_tile = self.available_tiles[self.active_tile_idx]
        color = self.TILE_COLORS[active_tile]
        rect = pygame.Rect(preview_x - 15, preview_y - 15, 30, 30)
        self._draw_glowing_rect(self.screen, color, rect, 10)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - i / glow_size))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius + i, glow_color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def _draw_glowing_rect(self, surface, color, rect, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = int(80 * (1 - i / glow_size))
            glow_color = (*color, alpha)
            glow_rect = rect.inflate(i*2, i*2)
            
            temp_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, glow_color, (0, 0, *glow_rect.size), border_radius=5)
            surface.blit(temp_surf, glow_rect.topleft)
            
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class EffectParticle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.pos = [x, y]
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.life = random.randint(15, 30)
        self.max_life = self.life
        self.color = color
        self.radius = random.randint(2, 4)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.98
        self.vel[1] *= 0.98
        self.life -= 1

    def draw(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color, alpha)
        
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
        screen.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

# Example of how to run the environment
if __name__ == '__main__':
    # Set a non-dummy driver for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human interaction
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Tiles")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # move=none, space=released, shift=released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Wins: {info['consecutive_wins']}")
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    env.close()