
# Generated: 2025-08-27T22:26:27.262145
# Source Brief: brief_03124.md
# Brief Index: 3124

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the falling piece, ↑↓ to rotate it. "
        "Match 3 or more crystals of the same color to clear them."
    )

    game_description = (
        "Maneuver falling crystal pairs in a cavern to create color matches. "
        "Clear crystals to score points before the time runs out or the cavern fills up."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 12
    GAME_DURATION_STEPS = 1800  # 60 seconds * 30 FPS

    # --- Colors ---
    COLOR_BG = (25, 20, 35)
    COLOR_GRID = (45, 40, 55)
    CRYSTAL_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255),  # Blue
        4: (255, 255, 80),  # Yellow
    }
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_SHADOW = (10, 10, 10)
    COLOR_HIGHLIGHT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Iso projection helpers
        self.tile_width_half = 20
        self.tile_height_half = 10
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80
        
        self.grid = []
        self.active_piece = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_counter = 0
        self.fall_speed = 15 # Ticks per grid unit fall
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.fall_counter = 0

        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = self.game_over

        if not terminated:
            # 1. Handle player input
            move_success = self._handle_input(movement)
            if move_success:
                reward += 0.01 # Small reward for valid movement

            # 2. Update game logic (automatic fall)
            self.fall_counter += 1
            if self.fall_counter >= self.fall_speed:
                self.fall_counter = 0
                if not self._move_active_piece(0, 1):
                    # Piece has landed
                    self._land_piece()
                    
                    # Check for matches and chain reactions
                    combo_multiplier = 1
                    while True:
                        matches_found, match_reward = self._find_and_clear_matches()
                        if matches_found:
                            # sfx: match_clear.wav
                            reward += match_reward * combo_multiplier
                            self.score += int(match_reward * combo_multiplier * 10)
                            combo_multiplier += 1
                            self._apply_gravity()
                            # sfx: crystals_fall.wav
                        else:
                            break
                    
                    if self.game_over:
                        # Game over from stack reaching top
                        terminated = True
                    else:
                        self._spawn_new_piece()
                        if self.game_over: # Spawn failed
                            terminated = True


        # 3. Update step counter and check for time-based termination
        self.steps += 1
        if self.steps >= self.GAME_DURATION_STEPS:
            terminated = True
            if not self.game_over: # Avoid double penalty
                reward -= 10 # Penalty for running out of time

        self.game_over = terminated

        # 4. Update visual effects
        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement):
        # movement: 0=none, 1=up(rot_cw), 2=down(rot_ccw), 3=left, 4=right
        if self.active_piece is None:
            return False

        if movement == 1: # Rotate CW
            return self._rotate_active_piece(1)
        elif movement == 2: # Rotate CCW
            return self._rotate_active_piece(-1)
        elif movement == 3: # Move Left
            return self._move_active_piece(-1, 0)
        elif movement == 4: # Move Right
            return self._move_active_piece(1, 0)
        return False

    def _spawn_new_piece(self):
        # sfx: piece_spawn.wav
        self.active_piece = {
            "pos": [self.GRID_WIDTH // 2 - 1, 0],
            "colors": [self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1), 
                       self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)],
            "orientation": 0, # 0: vert (main top), 1: horiz (main left), 2: vert (main bot), 3: horiz (main right)
        }
        if not self._is_valid_position(self.active_piece):
            self.game_over = True

    def _get_piece_coords(self, piece):
        x, y = piece["pos"]
        o = piece["orientation"]
        if o == 0: return [(x, y), (x, y + 1)]
        if o == 1: return [(x, y), (x + 1, y)]
        if o == 2: return [(x, y), (x, y - 1)]
        if o == 3: return [(x, y), (x - 1, y)]
        return []

    def _is_valid_position(self, piece):
        coords = self._get_piece_coords(piece)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if self.grid[y][x] is not None:
                return False
        return True

    def _move_active_piece(self, dx, dy):
        if self.active_piece is None: return False
        
        new_piece = self.active_piece.copy()
        new_piece["pos"] = [self.active_piece["pos"][0] + dx, self.active_piece["pos"][1] + dy]
        
        if self._is_valid_position(new_piece):
            self.active_piece = new_piece
            return True
        return False

    def _rotate_active_piece(self, direction):
        if self.active_piece is None: return False

        # sfx: rotate.wav
        new_piece = self.active_piece.copy()
        new_piece["orientation"] = (self.active_piece["orientation"] + direction) % 4
        
        if self._is_valid_position(new_piece):
            self.active_piece = new_piece
            return True
        
        # Wall kick: try moving left/right
        for kick in [-1, 1]:
            kicked_piece = new_piece.copy()
            kicked_piece["pos"] = [new_piece["pos"][0] + kick, new_piece["pos"][1]]
            if self._is_valid_position(kicked_piece):
                self.active_piece = kicked_piece
                return True
        return False
    
    def _land_piece(self):
        if self.active_piece is None: return
        # sfx: piece_land.wav
        coords = self._get_piece_coords(self.active_piece)
        colors = self.active_piece["colors"]
        
        # The two crystals might not be in drawing order
        if coords[0][1] > coords[1][1] or (coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]):
            coords.reverse()
            
        for i, (x, y) in enumerate(coords):
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[y][x] = {"color": colors[i]}
                if y < 1: # Check if landing in the "danger zone" at the top
                    self.game_over = True
            else: # Piece landed out of bounds (should not happen with valid moves)
                self.game_over = True
        
        self.active_piece = None

    def _find_and_clear_matches(self):
        to_clear = set()
        visited = set()
        reward = 0
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] and (x, y) not in visited:
                    color = self.grid[y][x]["color"]
                    component = set()
                    q = deque([(x, y)])
                    
                    while q:
                        cx, cy = q.popleft()
                        if (cx, cy) in visited or not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
                            continue
                        if self.grid[cy][cx] and self.grid[cy][cx]["color"] == color:
                            visited.add((cx, cy))
                            component.add((cx, cy))
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                q.append((cx + dx, cy + dy))
                    
                    if len(component) >= 3:
                        to_clear.update(component)
                        if len(component) == 3: reward += 1
                        elif len(component) == 4: reward += 2
                        else: reward += 5

        if not to_clear:
            return False, 0

        for x, y in to_clear:
            self._create_particles(x, y, self.grid[y][x]["color"])
            self.grid[y][x] = None
        
        return True, reward

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is not None:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = None
                    empty_row -= 1

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.tile_width_half
        screen_y = self.origin_y + (x + y) * self.tile_height_half
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
            
        # Render landed crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x]:
                    self._render_crystal(x, y, self.grid[y][x]["color"])

        # Render active piece
        if self.active_piece:
            coords = self._get_piece_coords(self.active_piece)
            colors = self.active_piece["colors"]
            for i, (x, y) in enumerate(coords):
                self._render_crystal(x, y, colors[i], highlighted=True)
        
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_crystal(self, x, y, color_idx, highlighted=False):
        center_x, center_y = self._iso_to_screen(x, y)
        center_y += self.tile_height_half # Adjust to draw from top corner
        
        w = self.tile_width_half
        h = self.tile_height_half
        
        points = [
            (center_x, center_y - h),          # Top
            (center_x + w, center_y),          # Right
            (center_x, center_y + h),          # Bottom
            (center_x - w, center_y),          # Left
        ]

        base_color = self.CRYSTAL_COLORS[color_idx]
        light_color = tuple(min(255, c + 40) for c in base_color)
        dark_color = tuple(max(0, c - 40) for c in base_color)
        
        # Draw faces
        pygame.gfxdraw.filled_polygon(self.screen, [(points[0]), (points[1]), (center_x, center_y)], light_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(points[1]), (points[2]), (center_x, center_y)], base_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(points[2]), (points[3]), (center_x, center_y)], dark_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(points[3]), (points[0]), (center_x, center_y)], base_color)
        
        # Draw outlines
        pygame.gfxdraw.aapolygon(self.screen, points, light_color)
        
        if highlighted:
            glow_points = [
                (center_x, center_y - h - 2),
                (center_x + w + 2, center_y),
                (center_x, center_y + h + 2),
                (center_x - w - 2, center_y),
            ]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_HIGHLIGHT)

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        time_left = max(0, (self.GAME_DURATION_STEPS - self.steps) / 30)
        time_text = f"Time: {int(time_left)}"
        
        self._draw_text(score_text, (20, 20), self.font_large)
        self._draw_text(time_text, (self.WIDTH - 150, 20), self.font_large)
        
        if self.game_over:
            end_text = "GAME OVER"
            self._draw_text(end_text, (self.WIDTH//2, self.HEIGHT//2 - 30), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, shadow_color=COLOR_UI_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, color_idx):
        screen_x, screen_y = self._iso_to_screen(grid_x, grid_y)
        base_color = self.CRYSTAL_COLORS[color_idx]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': screen_x,
                'y': screen_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 1, # Bias upwards
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': tuple(min(255, c + self.np_random.integers(-20, 20)) for c in base_color)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['size'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("      CRYSTAL CAVERNS")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to human screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # Transpose back from (H, W, C) to (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Control human play speed

    print(f"Game Over! Final Score: {info['score']}")
    env.close()