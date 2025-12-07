import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:36:18.892805
# Source Brief: brief_01087.md
# Brief Index: 1087
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for explosion effects."""
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 6)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = random.randint(20, 40)
        self.radius = random.uniform(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius -= 0.1
        self.vx *= 0.95
        self.vy *= 0.95

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifetime / 40))))
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.radius), int(self.radius)), int(self.radius))
            surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Drop colored tiles into the grid to match three or more of the same color. "
        "Create chain reactions to score big before time runs out!"
    )
    user_guide = (
        "Controls: ←→ to move the selector, space to drop a tile, and shift to cycle the tile color."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 5, 5
    MAX_STEPS = 1800  # Approx 60 seconds at 30 FPS
    WIN_SCORE = 500
    CELL_SIZE = 60
    GRID_LINE_WIDTH = 2
    TILE_BORDER_RADIUS = 8

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 200, 0)
    TILE_COLORS = {
        1: ((220, 50, 50), (255, 100, 100)),  # Red (base, highlight)
        2: ((50, 220, 50), (100, 255, 100)),  # Green
        3: ((50, 50, 220), (100, 100, 255)),  # Blue
    }
    COLOR_WHITE = (255, 255, 255)

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
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)

        self.grid_x = (self.SCREEN_WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.grid_y = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.CELL_SIZE) // 2

        # self.reset() is called by the wrapper or user, not in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.selector_col = self.GRID_COLS // 2
        self.current_tile_color = 1

        self.prev_movement = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.falling_tiles = []
        self.flash_effects = [] # (r, c, lifetime)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        placed_tile = False

        # --- Handle Input ---
        if movement in [3, 4] and movement != self.prev_movement: # Left/Right Tap
            if movement == 3: self.selector_col = max(0, self.selector_col - 1)
            if movement == 4: self.selector_col = min(self.GRID_COLS - 1, self.selector_col + 1)
        self.prev_movement = movement

        if shift_held and not self.prev_shift_held: # Shift Tap
            self.current_tile_color = (self.current_tile_color % 3) + 1
            # sfx: color_cycle.wav
        self.prev_shift_held = shift_held

        if space_held and not self.prev_space_held: # Space Tap
            if self.grid[0, self.selector_col] == 0:
                placed_tile = self._place_tile()
                reward += 0.1 # Small reward for a valid action
                # sfx: tile_place.wav
        self.prev_space_held = space_held

        # --- Game Logic ---
        exploded_tiles_count = 0
        if placed_tile:
            exploded_tiles_count = self._process_chain_reactions()

        if exploded_tiles_count > 0:
            reward += exploded_tiles_count # +1 per exploded tile

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 10

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tile(self):
        for r in range(self.GRID_ROWS - 1, -1, -1):
            if self.grid[r, self.selector_col] == 0:
                self.grid[r, self.selector_col] = self.current_tile_color
                return True
        return False

    def _process_chain_reactions(self):
        total_exploded = 0
        chain_level = 1
        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_exploded = len(matches)
            total_exploded += num_exploded
            self.score += num_exploded * int(10 * chain_level)
            # sfx: explosion_chain.wav

            for r, c in matches:
                self._create_particles(r, c, self.grid[r, c])
                self.flash_effects.append((r, c, 5))
                self.grid[r, c] = 0

            self._apply_gravity()
            chain_level += 1
        return total_exploded

    def _find_matches(self):
        to_explode = set()
        visited = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != 0 and not visited[r, c]:
                    color = self.grid[r, c]
                    component = set()
                    q = [(r, c)]
                    visited[r, c] = True
                    
                    head = 0
                    while head < len(q):
                        curr_r, curr_c = q[head]
                        head += 1
                        component.add((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                               not visited[nr, nc] and self.grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    
                    if len(component) >= 3:
                        to_explode.update(component)
        return to_explode

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        start_pos = self._grid_to_pixel(r, c)
                        end_pos = self._grid_to_pixel(empty_row, c)
                        self.falling_tiles.append({
                            'color_idx': self.grid[r, c],
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'progress': 0.0
                        })
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        # Check if any top row cell is blocked
        if any(self.grid[0, c] != 0 for c in range(self.GRID_COLS)):
            self.game_over = True
            self.win = False
            return True
        return False

    def _get_observation(self):
        self._update_animations()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "win": self.win}

    def _grid_to_pixel(self, r, c):
        x = self.grid_x + c * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.grid_y + r * self.CELL_SIZE + self.CELL_SIZE / 2
        return x, y

    def _update_animations(self):
        # Update falling tiles
        for tile in self.falling_tiles[:]:
            tile['progress'] += 0.15 # Animation speed
            if tile['progress'] >= 1.0:
                self.falling_tiles.remove(tile)
        
        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

        # Update flash effects
        new_flash_effects = []
        for r, c, lifetime in self.flash_effects:
            lifetime -= 1
            if lifetime > 0:
                new_flash_effects.append((r, c, lifetime))
        self.flash_effects = new_flash_effects

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.grid_y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_x, y), (self.grid_x + self.GRID_COLS * self.CELL_SIZE, y), self.GRID_LINE_WIDTH)
        for c in range(self.GRID_COLS + 1):
            x = self.grid_x + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_y), (x, self.grid_y + self.GRID_ROWS * self.CELL_SIZE), self.GRID_LINE_WIDTH)

        # Draw static tiles
        rendered_by_fall = set()
        for tile in self.falling_tiles:
            start_x, _ = tile['start_pos']
            col = int((start_x - self.grid_x - self.CELL_SIZE / 2) / self.CELL_SIZE + 0.5)
            for r in range(self.GRID_ROWS):
                rendered_by_fall.add((r, col))

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != 0 and (r,c) not in rendered_by_fall:
                    self._draw_tile(r, c, self.grid[r, c])
        
        # Draw falling tiles
        for tile in self.falling_tiles:
            progress = min(1.0, tile['progress'])
            # Ease out cubic interpolation
            interp = 1 - pow(1 - progress, 3)
            
            start_x, start_y = tile['start_pos']
            end_x, end_y = tile['end_pos']
            x = start_x + (end_x - start_x) * interp
            y = start_y + (end_y - start_y) * interp
            self._draw_tile_at_pixel(x, y, tile['color_idx'])

        # Draw flash effects
        for r, c, lifetime in self.flash_effects:
            alpha = int(255 * (lifetime / 5))
            px, py = self._grid_to_pixel(r, c)
            size = self.CELL_SIZE * 0.9
            flash_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surf, (px - size/2, py - size/2))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_tile(self, r, c, color_idx):
        x, y = self._grid_to_pixel(r, c)
        self._draw_tile_at_pixel(x, y, color_idx)
    
    def _draw_tile_at_pixel(self, x, y, color_idx):
        size = self.CELL_SIZE * 0.9
        rect = pygame.Rect(x - size / 2, y - size / 2, size, size)
        base_color, highlight_color = self.TILE_COLORS[color_idx]
        pygame.draw.rect(self.screen, base_color, rect, border_radius=self.TILE_BORDER_RADIUS)
        pygame.draw.rect(self.screen, highlight_color, rect, width=3, border_radius=self.TILE_BORDER_RADIUS)

    def _render_ui(self):
        # Draw Selector
        sel_x = self.grid_x + self.selector_col * self.CELL_SIZE + self.CELL_SIZE / 2
        sel_y = self.grid_y - 15
        points = [(sel_x - 10, sel_y), (sel_x + 10, sel_y), (sel_x, sel_y + 10)]
        pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, points)

        # Draw Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Draw Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_sec = time_left / 30.0
        timer_text = self.font_main.render(f"TIME: {time_sec:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 20))
        
        # Draw Next Tile Preview
        preview_text = self.font_main.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(preview_text, (20, self.SCREEN_HEIGHT - 50))
        self._draw_tile_at_pixel(120, self.SCREEN_HEIGHT - 35, self.current_tile_color)

        # Draw Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.win else "GAME OVER"
            status_render = self.font_large.render(status_text, True, self.COLOR_WHITE)
            status_rect = status_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(status_render, status_rect)

            final_score_render = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(final_score_render, final_score_rect)

    def _create_particles(self, r, c, color_idx):
        px, py = self._grid_to_pixel(r, c)
        base_color, _ = self.TILE_COLORS[color_idx]
        for _ in range(20):
            # Mix of tile color and white
            mix_ratio = random.random()
            p_color = (
                int(base_color[0] * mix_ratio + self.COLOR_WHITE[0] * (1 - mix_ratio)),
                int(base_color[1] * mix_ratio + self.COLOR_WHITE[1] * (1 - mix_ratio)),
                int(base_color[2] * mix_ratio + self.COLOR_WHITE[2] * (1 - mix_ratio)),
            )
            self.particles.append(Particle(px, py, p_color))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and will not be run by the test suite.
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Tile Placer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keyboard to MultiDiscrete action space
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()