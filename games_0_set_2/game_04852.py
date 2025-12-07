
# Generated: 2025-08-28T03:12:36.496823
# Source Brief: brief_04852.md
# Brief Index: 4852

        
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

    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop, ↑ for hard drop. Space/Shift to rotate."
    )

    game_description = (
        "Fast-paced isometric puzzle game. Rotate and place falling blocks to clear lines and score points."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 22  # 20 visible, 2 hidden at top for spawning
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 10000

    # Visuals
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_WIDTH, TILE_HEIGHT = 28, 14
    TILE_DEPTH = 14
    
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_DANGER_ZONE = (60, 40, 40)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GHOST = (255, 255, 255, 40)

    # Tetromino shapes (I, O, T, L, J, S, Z) and their colors
    TETROMINOES = [
        {"shape": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], "color": (0, 240, 240)},  # I
        {"shape": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], "color": (240, 240, 0)},  # O
        {"shape": [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)], "color": (160, 0, 240)},  # T
        {"shape": [(2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)], "color": (240, 160, 0)},  # L
        {"shape": [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)], "color": (0, 0, 240)},  # J
        {"shape": [(1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0)], "color": (0, 240, 0)},    # S
        {"shape": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)], "color": (240, 0, 0)},    # Z
    ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 60

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.base_fall_speed = 0.02
        self.current_fall_speed = self.base_fall_speed
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        self.line_clear_animation = []

        self._spawn_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for every step
        self.steps += 1
        
        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward -= 100
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle rotation (on button press, not hold)
        if space_held and not self.prev_space_held:
            self._rotate_piece(1)  # Clockwise
        if shift_held and not self.prev_shift_held:
            self._rotate_piece(-1) # Counter-clockwise

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Handle movement
        if movement == 1: # Up -> Hard Drop
            while self._is_valid_position(self.piece['pos']):
                self.piece['pos'] = (self.piece['pos'][0], self.piece['pos'][1] + 1)
            self.piece['pos'] = (self.piece['pos'][0], self.piece['pos'][1] - 1)
            self.piece['fall_progress'] = 1.0 # Force placement on next update
        elif movement == 2: # Down -> Soft Drop
            self.piece['fall_progress'] += self.current_fall_speed * 5
        elif movement == 3: # Left
            self._move_piece(-1, 0)
        elif movement == 4: # Right
            self._move_piece(1, 0)

    def _update_game_state(self):
        reward = 0
        
        # Update particles and animations
        self._update_particles()
        if self.line_clear_animation and self.line_clear_animation[0]['timer'] > 0:
            self.line_clear_animation[0]['timer'] -= 1
            return 0 # Pause game during animation
        elif self.line_clear_animation:
            self.line_clear_animation.pop(0)

        # Piece gravity
        self.piece['fall_progress'] += self.current_fall_speed
        if self.piece['fall_progress'] >= 1.0:
            self.piece['fall_progress'] = 0.0
            
            next_pos = (self.piece['pos'][0], self.piece['pos'][1] + 1)
            if self._is_valid_position(next_pos):
                self.piece['pos'] = next_pos
            else:
                # Place piece and get reward
                self._place_piece()
                reward += 0.1
                
                # Check for line clears
                lines_cleared_count, lines = self._check_clear_lines()
                if lines_cleared_count > 0:
                    self.lines_cleared += lines_cleared_count
                    
                    # Calculate score and reward
                    base_score = [0, 100, 300, 500, 800][lines_cleared_count]
                    self.score += base_score
                    reward += lines_cleared_count + (2 if lines_cleared_count > 1 else 0)

                    self._start_line_clear_animation(lines)
                    # Increase difficulty
                    self.current_fall_speed = self.base_fall_speed + self.lines_cleared * 0.005
                
                # Spawn new piece and check for game over
                self._spawn_piece()
                if not self._is_valid_position(self.piece['pos']):
                    self.game_over = True
        
        return reward

    def _spawn_piece(self):
        shape_idx = self.np_random.integers(0, len(self.TETROMINOES))
        self.piece = {
            'shape_idx': shape_idx,
            'rotation': 0,
            'pos': (self.GRID_WIDTH // 2 - 2, 0),
            'fall_progress': 0.0,
            'shape': self.TETROMINOES[shape_idx]['shape'],
            'color': self.TETROMINOES[shape_idx]['color']
        }
        
    def _get_rotated_shape(self, shape, rotation):
        rotated_coords = []
        for x, y, z in shape:
            if rotation == 0:
                rotated_coords.append((x, y, z))
            elif rotation == 1:
                rotated_coords.append((2 - y, x, z))
            elif rotation == 2:
                rotated_coords.append((2 - x, 2 - y, z))
            elif rotation == 3:
                rotated_coords.append((y, 2 - x, z))
        return rotated_coords

    def _is_valid_position(self, pos, rotation=None):
        if rotation is None:
            rotation = self.piece['rotation']
        
        shape_coords = self._get_rotated_shape(self.piece['shape'], rotation)

        for x, y, z in shape_coords:
            grid_x = pos[0] + x
            grid_y = pos[1] + y
            
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return False
            if self.grid[grid_y, grid_x] != 0:
                return False
        return True

    def _move_piece(self, dx, dy):
        new_pos = (self.piece['pos'][0] + dx, self.piece['pos'][1] + dy)
        if self._is_valid_position(new_pos):
            self.piece['pos'] = new_pos
            return True
        return False
        
    def _rotate_piece(self, direction):
        # sfx: rotate_sound
        new_rotation = (self.piece['rotation'] + direction) % 4
        
        # Wall Kick logic (SRS-like)
        test_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for ox, oy in test_offsets:
            new_pos = (self.piece['pos'][0] + ox, self.piece['pos'][1] + oy)
            if self._is_valid_position(new_pos, new_rotation):
                self.piece['rotation'] = new_rotation
                self.piece['pos'] = new_pos
                return

    def _place_piece(self):
        # sfx: place_block_sound
        shape_coords = self._get_rotated_shape(self.piece['shape'], self.piece['rotation'])
        for x, y, z in shape_coords:
            grid_x = self.piece['pos'][0] + x
            grid_y = self.piece['pos'][1] + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y, grid_x] = self.piece['shape_idx'] + 1
                
                # Add placement particles
                iso_x, iso_y = self._to_iso(grid_x, grid_y, 0)
                for _ in range(3):
                    self.particles.append(self._create_particle(iso_x, iso_y, self.piece['color']))

    def _check_clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # sfx: line_clear_sound
            return len(lines_to_clear), lines_to_clear
        return 0, []

    def _start_line_clear_animation(self, lines):
        self.line_clear_animation.append({'lines': lines, 'timer': 10}) # 10 frames flash
        for r in lines:
            for c in range(self.GRID_WIDTH):
                iso_x, iso_y = self._to_iso(c, r, 0)
                for _ in range(5):
                    self.particles.append(self._create_particle(iso_x, iso_y, (255, 255, 255), 30))

        # We defer the actual clearing until after the animation
        def clear_and_drop():
            new_grid = np.zeros_like(self.grid)
            new_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if r not in lines:
                    new_grid[new_row, :] = self.grid[r, :]
                    new_row -= 1
            self.grid = new_grid
        
        self.line_clear_animation.append({'action': clear_and_drop, 'timer': 1})

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
            "lines": self.lines_cleared,
        }

    # --- Rendering Methods ---

    def _to_iso(self, x, y, z):
        iso_x = self.grid_origin_x + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.grid_origin_y + (x + y) * (self.TILE_HEIGHT / 2) - z * self.TILE_DEPTH
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, z, color, alpha=255):
        iso_x, iso_y = self._to_iso(x, y, z)
        
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)

        points_top = [
            (iso_x, iso_y),
            (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x - self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2)
        ]
        points_left = [
            (iso_x - self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x, iso_y + self.TILE_HEIGHT + self.TILE_DEPTH),
            (iso_x - self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2 + self.TILE_DEPTH)
        ]
        points_right = [
            (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x, iso_y + self.TILE_HEIGHT + self.TILE_DEPTH),
            (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2 + self.TILE_DEPTH)
        ]
        
        pygame.gfxdraw.filled_polygon(surface, points_top, (*color, alpha))
        pygame.gfxdraw.filled_polygon(surface, points_left, (*dark_color, alpha))
        pygame.gfxdraw.filled_polygon(surface, points_right, (*light_color, alpha))
        
        pygame.gfxdraw.aapolygon(surface, points_top, (*color, alpha))
        pygame.gfxdraw.aapolygon(surface, points_left, (*dark_color, alpha))
        pygame.gfxdraw.aapolygon(surface, points_right, (*light_color, alpha))

    def _render_game(self):
        # Draw grid lines and danger zone
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                iso_x, iso_y = self._to_iso(x, y, 0)
                points = [
                    (iso_x, iso_y),
                    (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2),
                    (iso_x, iso_y + self.TILE_HEIGHT),
                    (iso_x - self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2)
                ]
                color = self.COLOR_DANGER_ZONE if y < 2 else self.COLOR_GRID
                pygame.gfxdraw.aapolygon(self.screen, points, color)
        
        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color_idx = self.grid[r, c] - 1
                    color = self.TETROMINOES[color_idx]['color']
                    
                    # Flashing animation for line clears
                    is_flashing = False
                    if self.line_clear_animation and self.line_clear_animation[0]['timer'] > 0:
                         if r in self.line_clear_animation[0]['lines']:
                             is_flashing = True
                             flash_state = (self.line_clear_animation[0]['timer'] // 2) % 2
                             color = (255, 255, 255) if flash_state == 1 else color

                    if not is_flashing:
                        self._draw_iso_cube(self.screen, c, r, 0, color)

        # Draw ghost piece
        if not self.game_over:
            ghost_pos = self.piece['pos']
            while self._is_valid_position((ghost_pos[0], ghost_pos[1] + 1)):
                ghost_pos = (ghost_pos[0], ghost_pos[1] + 1)
            
            shape_coords = self._get_rotated_shape(self.piece['shape'], self.piece['rotation'])
            for x, y, z in shape_coords:
                self._draw_iso_cube(self.screen, ghost_pos[0] + x, ghost_pos[1] + y, 0, self.COLOR_GHOST[:3], self.COLOR_GHOST[3])

        # Draw falling piece
        if not self.game_over:
            shape_coords = self._get_rotated_shape(self.piece['shape'], self.piece['rotation'])
            fall_offset = self.piece['fall_progress']
            for x, y, z in shape_coords:
                self._draw_iso_cube(self.screen, self.piece['pos'][0] + x, self.piece['pos'][1] + y + fall_offset, 0, self.piece['color'])

        # Draw particles
        self._draw_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_text = self.font_small.render(f"LINES: {self.lines_cleared} / {self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (20, 55))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "GAME OVER"
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                end_text = "YOU WIN!"
            
            text_surface = self.font_main.render(end_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    # --- Particle System ---
    def _create_particle(self, x, y, color, lifetime=20):
        return {
            'x': x, 'y': y, 'vx': self.np_random.uniform(-1, 1), 'vy': self.np_random.uniform(-2, 0),
            'lifetime': lifetime, 'max_lifetime': lifetime, 'color': color
        }

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            size = max(1, int(3 * (p['lifetime'] / p['max_lifetime'])))
            pygame.draw.circle(self.screen, color, (int(p['x']), int(p['y'])), size)

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# This block allows you to run the game directly for testing
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Ensure it runs headlessly if needed
    
    # For interactive testing, comment out the line above and uncomment the block below
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Interactive Play ---
    # Re-initialize with a real display
    pygame.display.set_caption("Isometric Puzzle")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the actual screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    pygame.quit()
    print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines']}")