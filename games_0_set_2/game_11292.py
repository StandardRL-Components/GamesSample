import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:45:27.000695
# Source Brief: brief_01292.md
# Brief Index: 1292
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An underwater-themed puzzle game where the player must match pairs of
    glowing glyphs before time runs out. The game is designed as a
    Gymnasium environment with a focus on high-quality visuals and engaging gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "An underwater-themed puzzle game. Match pairs of glowing glyphs on a grid before the timer runs out."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select a tile. Press shift to clear your current selection."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 2000
    MAX_LEVELS = 5

    # --- Colors ---
    COLOR_BG_DEEP = (5, 10, 25)
    COLOR_BG_SHALLOW = (10, 25, 60)
    COLOR_RUINS = (20, 35, 70)
    COLOR_TILE_HIDDEN = (30, 50, 90)
    COLOR_TILE_BORDER = (50, 80, 140)
    COLOR_GLYPH_LOCKED = (60, 90, 150)
    COLOR_GOLD = (255, 215, 0)
    COLOR_GREEN = (50, 255, 50)
    COLOR_RED = (255, 50, 50)
    COLOR_WHITE = (240, 240, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (200, 220, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.tiles = []
        self.grid_layout = (0, 0)
        self.grid_rect = pygame.Rect(0, 0, 0, 0)
        self.cursor_logical_pos = (0, 0)
        self.cursor_visual_pos = [0.0, 0.0]
        self.selected_indices = []
        self.mismatch_timer = 0
        self.time_remaining = 0
        self.max_time = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.bubbles = []
        self.last_reward_info = {"type": "None", "value": 0}

        # The reset call is needed to initialize the np_random generator
        # before it might be used in other setup methods.
        self.np_random = None # Will be set by super().reset()
        self.reset()
        
        # --- Critical Self-Check ---
        # This can be commented out for production, but is useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles.clear()
        
        self._setup_bubbles()
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        self.last_reward_info = {"type": "Time", "value": -0.01} # Small penalty for taking time
        reward += self.last_reward_info["value"]

        # --- Update Timers ---
        self.time_remaining -= 1.0 / self.FPS
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                self._revert_mismatched_tiles()

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_movement(movement)
        
        # Rising edge detection for space and shift
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        if shift_press:
            reward += self._handle_deselect()
        
        if space_press:
            reward += self._handle_selection()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Check for level completion ---
        if all(tile['state'] == 'matched' for tile in self.tiles):
            # sfx: level_complete
            reward += 5.0
            self.last_reward_info = {"type": "Level Up", "value": 5.0}
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.game_over = True
            else:
                self._setup_level()

        # --- Check for Termination ---
        terminated = self.game_over
        if self.time_remaining <= 0:
            # sfx: game_over_timeout
            reward -= 100.0
            self.last_reward_info = {"type": "Timeout", "value": -100.0}
            terminated = True
        elif self.level > self.MAX_LEVELS:
            # sfx: victory
            reward += 100.0
            self.last_reward_info = {"type": "Victory", "value": 100.0}
            terminated = True
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._update_and_draw_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining,
            "last_reward_type": self.last_reward_info["type"],
        }

    # --- Game Logic Helpers ---
    def _setup_level(self):
        self.selected_indices.clear()
        self.mismatch_timer = 0
        num_tiles = 4 + (self.level - 1) * 2
        self.grid_layout = self._get_grid_layout(num_tiles)
        
        cols, rows = self.grid_layout
        tile_values = list(range(num_tiles // 2)) * 2
        self.np_random.shuffle(tile_values)

        grid_width = cols * 80 + (cols - 1) * 10
        grid_height = rows * 80 + (rows - 1) * 10
        start_x = (self.SCREEN_WIDTH - grid_width) // 2
        start_y = (self.SCREEN_HEIGHT - grid_height) // 2 + 20
        self.grid_rect = pygame.Rect(start_x - 20, start_y - 20, grid_width + 40, grid_height + 40)

        self.tiles.clear()
        for i in range(num_tiles):
            row, col = i // cols, i % cols
            tile_rect = pygame.Rect(
                start_x + col * 90,
                start_y + row * 90,
                80, 80
            )
            self.tiles.append({
                'value': tile_values[i],
                'rect': tile_rect,
                'state': 'hidden', # hidden, selected, matched
                'flip_progress': 0.0,
                'target_flip': 0.0
            })
        
        self.cursor_logical_pos = (0, 0)
        tile_under_cursor = self.tiles[0]
        self.cursor_visual_pos = [float(tile_under_cursor['rect'].centerx), float(tile_under_cursor['rect'].centery)]
        
        self.max_time = max(30, 60 - (self.level - 1) * 5)
        self.time_remaining = self.max_time

    def _handle_movement(self, movement):
        row, col = self.cursor_logical_pos
        cols, rows = self.grid_layout
        if movement == 1 and row > 0: # Up
            self.cursor_logical_pos = (row - 1, col)
        elif movement == 2 and row < rows - 1: # Down
            self.cursor_logical_pos = (row + 1, col)
        elif movement == 3 and col > 0: # Left
            self.cursor_logical_pos = (row, col - 1)
        elif movement == 4 and col < cols - 1: # Right
            self.cursor_logical_pos = (row, col + 1)

    def _handle_selection(self):
        if self.mismatch_timer > 0 or len(self.selected_indices) >= 2:
            return 0.0

        row, col = self.cursor_logical_pos
        cols, _ = self.grid_layout
        index = row * cols + col
        
        if self.tiles[index]['state'] == 'hidden':
            # sfx: tile_select
            self.tiles[index]['state'] = 'selected'
            self.tiles[index]['target_flip'] = 1.0
            self.selected_indices.append(index)
            
            if len(self.selected_indices) == 2:
                return self._check_match()
        return 0.0

    def _handle_deselect(self):
        if self.selected_indices:
            # sfx: deselect
            for index in self.selected_indices:
                self.tiles[index]['state'] = 'hidden'
                self.tiles[index]['target_flip'] = 0.0
            self.selected_indices.clear()
            self.last_reward_info = {"type": "Deselect", "value": -0.05}
            return -0.05
        return 0.0
        
    def _check_match(self):
        idx1, idx2 = self.selected_indices
        tile1, tile2 = self.tiles[idx1], self.tiles[idx2]

        if tile1['value'] == tile2['value']:
            # sfx: match_success
            tile1['state'] = 'matched'
            tile2['state'] = 'matched'
            self.selected_indices.clear()
            self.time_remaining = min(self.max_time, self.time_remaining + 2)
            self._create_particles(tile1['rect'].center, self.COLOR_GREEN, 30)
            self._create_particles(tile2['rect'].center, self.COLOR_GREEN, 30)
            self.last_reward_info = {"type": "Match", "value": 1.0}
            return 1.0
        else:
            # sfx: match_fail
            self.mismatch_timer = self.FPS // 2 # 0.5 second delay
            self.time_remaining -= 5
            self._create_particles(tile1['rect'].center, self.COLOR_RED, 20)
            self._create_particles(tile2['rect'].center, self.COLOR_RED, 20)
            self.last_reward_info = {"type": "Mismatch", "value": -0.1}
            return -0.1

    def _revert_mismatched_tiles(self):
        for index in self.selected_indices:
            if self.tiles[index]['state'] != 'matched':
                self.tiles[index]['state'] = 'hidden'
                self.tiles[index]['target_flip'] = 0.0
        self.selected_indices.clear()

    # --- Rendering Helpers ---
    def _update_and_draw_frame(self):
        self._draw_background()
        self._update_and_draw_bubbles()
        self._update_and_draw_particles()
        
        # Draw grid container
        pygame.draw.rect(self.screen, self.COLOR_RUINS, self.grid_rect, border_radius=15)
        pygame.draw.rect(self.screen, self.COLOR_TILE_BORDER, self.grid_rect, width=3, border_radius=15)
        
        self._update_and_draw_tiles()
        self._update_and_draw_cursor()
        self._draw_ui()

        if self.game_over:
            self._draw_game_over_screen()

    def _draw_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_DEEP[0] * (1 - interp) + self.COLOR_BG_SHALLOW[0] * interp),
                int(self.COLOR_BG_DEEP[1] * (1 - interp) + self.COLOR_BG_SHALLOW[1] * interp),
                int(self.COLOR_BG_DEEP[2] * (1 - interp) + self.COLOR_BG_SHALLOW[2] * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        # Static "ruins"
        pygame.draw.rect(self.screen, self.COLOR_RUINS, (0, 350, 100, 50))
        pygame.draw.rect(self.screen, self.COLOR_RUINS, (580, 200, 60, 200))
        pygame.draw.circle(self.screen, self.COLOR_RUINS, (320, 450), 100)

    def _update_and_draw_tiles(self):
        for tile in self.tiles:
            # Animate flip
            if tile['flip_progress'] != tile['target_flip']:
                tile['flip_progress'] += (tile['target_flip'] - tile['flip_progress']) * 0.2
                if abs(tile['flip_progress'] - tile['target_flip']) < 0.01:
                    tile['flip_progress'] = tile['target_flip']
            
            # Draw tile based on state and flip progress
            r = tile['rect']
            center_x = r.centerx
            half_width = r.width / 2
            scale = abs(math.cos(tile['flip_progress'] * math.pi))
            current_width = int(r.width * scale)
            
            display_rect = pygame.Rect(center_x - current_width // 2, r.y, current_width, r.height)
            
            is_front_facing = tile['flip_progress'] > 0.5
            
            if is_front_facing:
                # Draw front (glyph)
                color = self.COLOR_GOLD if tile['state'] == 'matched' else self.COLOR_WHITE
                pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, display_rect, border_radius=8)
                self._draw_glyph(self.screen, tile['value'], display_rect, color)
            else:
                # Draw back
                pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, display_rect, border_radius=8)
                self._draw_glyph(self.screen, -1, display_rect, self.COLOR_GLYPH_LOCKED) # Back symbol

            # Draw border and glow
            if tile['state'] == 'selected':
                self._draw_glow(self.screen, r, self.COLOR_WHITE, 15, 5)
            elif tile['state'] == 'matched':
                self._draw_glow(self.screen, r, self.COLOR_GOLD, 10, 3)
            
            pygame.draw.rect(self.screen, self.COLOR_TILE_BORDER, r, width=2, border_radius=8)

    def _update_and_draw_cursor(self):
        row, col = self.cursor_logical_pos
        cols, _ = self.grid_layout
        target_idx = row * cols + col
        target_rect = self.tiles[target_idx]['rect']
        
        # Interpolate visual position for smooth movement
        self.cursor_visual_pos[0] += (target_rect.centerx - self.cursor_visual_pos[0]) * 0.4
        self.cursor_visual_pos[1] += (target_rect.centery - self.cursor_visual_pos[1]) * 0.4

        cursor_rect = pygame.Rect(0, 0, 90, 90)
        cursor_rect.center = (int(self.cursor_visual_pos[0]), int(self.cursor_visual_pos[1]))
        
        self._draw_glow(self.screen, cursor_rect, self.COLOR_CURSOR, 20, 5)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=12)

    def _draw_ui(self):
        # Score and Level
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_ui.render(f"GLYPH: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        # Time Bar
        time_ratio = max(0, self.time_remaining / self.max_time)
        bar_width = self.SCREEN_WIDTH - 200
        bar_fill = int(bar_width * time_ratio)
        bar_rect = pygame.Rect(100, 12, bar_width, 16)
        fill_rect = pygame.Rect(100, 12, bar_fill, 16)
        
        time_color = self.COLOR_GREEN
        if time_ratio < 0.5: time_color = self.COLOR_GOLD
        if time_ratio < 0.2: time_color = self.COLOR_RED
            
        pygame.draw.rect(self.screen, self.COLOR_BG_DEEP, bar_rect, border_radius=8)
        if bar_fill > 0:
            pygame.draw.rect(self.screen, time_color, fill_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bar_rect, width=2, border_radius=8)
        
    def _draw_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "VICTORY" if self.level > self.MAX_LEVELS else "TIME UP"
        text_surf = self.font_big.render(message, True, self.COLOR_GOLD if message == "VICTORY" else self.COLOR_RED)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)
        
        final_score_surf = self.font_ui.render(f"Final Score: {int(self.score)}", True, self.COLOR_WHITE)
        final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
        self.screen.blit(final_score_surf, final_score_rect)

    def _draw_glyph(self, surface, value, rect, color):
        cx, cy = rect.center
        w, h = rect.width, rect.height
        s = min(w, h) * 0.3 # Scale factor for glyph size

        if value == -1: # Back of tile
            pygame.draw.circle(surface, color, (cx, cy), s * 1.2, width=int(s/5))
            pygame.draw.circle(surface, color, (cx, cy), s * 0.5, width=int(s/5))
            return
            
        points = []
        if value == 0: # Triangle
            points = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
            pygame.draw.polygon(surface, color, points, width=int(s/4))
        elif value == 1: # Square
            pygame.draw.rect(surface, color, (cx - s*0.8, cy - s*0.8, s*1.6, s*1.6), width=int(s/4))
        elif value == 2: # X
            pygame.draw.line(surface, color, (cx - s, cy - s), (cx + s, cy + s), width=int(s/4))
            pygame.draw.line(surface, color, (cx + s, cy - s), (cx - s, cy + s), width=int(s/4))
        elif value == 3: # Circle
            pygame.draw.circle(surface, color, (cx, cy), s, width=int(s/4))
        elif value == 4: # Hourglass
            points = [(cx - s, cy - s), (cx + s, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
            pygame.draw.polygon(surface, color, points, width=int(s/4))
        elif value == 5: # Plus
            pygame.draw.line(surface, color, (cx, cy - s), (cx, cy + s), width=int(s/4))
            pygame.draw.line(surface, color, (cx - s, cy), (cx + s, cy), width=int(s/4))
        elif value == 6: # Diamond
            points = [(cx, cy - s*1.2), (cx - s*1.2, cy), (cx, cy + s*1.2), (cx + s*1.2, cy)]
            pygame.draw.polygon(surface, color, points, width=int(s/4))
        elif value == 7: # Wavy Line
            points = [(cx - s, cy - s/2), (cx - s/2, cy + s/2), (cx + s/2, cy - s/2), (cx + s, cy + s/2)]
            pygame.draw.lines(surface, color, False, points, width=int(s/4))
            
    def _draw_glow(self, surface, rect, color, radius, intensity):
        glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(intensity):
            alpha = 255 - (i * (255 // intensity))
            pygame.draw.circle(glow_surf, (*color, alpha // 4), (radius, radius), radius - i * (radius / intensity))
        
        scaled_glow = pygame.transform.smoothscale(glow_surf, (rect.width + radius, rect.height + radius))
        pos = (rect.centerx - scaled_glow.get_width()//2, rect.centery - scaled_glow.get_height()//2)
        surface.blit(scaled_glow, pos, special_flags=pygame.BLEND_RGBA_ADD)

    # --- Particle and Bubble System ---
    def _setup_bubbles(self):
        self.bubbles.clear()
        for _ in range(20):
            self.bubbles.append({
                'pos': [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                'radius': self.np_random.uniform(1, 4),
                'speed': self.np_random.uniform(0.2, 0.8)
            })

    def _update_and_draw_bubbles(self):
        for bubble in self.bubbles:
            bubble['pos'][1] -= bubble['speed']
            bubble['pos'][0] += math.sin(bubble['pos'][1] / 30) * 0.2
            if bubble['pos'][1] < -5:
                bubble['pos'] = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 5]
            
            color = (100, 150, 255, 50)
            pygame.gfxdraw.filled_circle(self.screen, int(bubble['pos'][0]), int(bubble['pos'][1]), int(bubble['radius']), color)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 31),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, p['pos'], special_flags=pygame.BLEND_RGBA_ADD)

    # --- Utility ---
    def _get_grid_layout(self, num_tiles):
        if num_tiles <= 0: return (0, 0)
        factors = []
        for i in range(1, int(math.sqrt(num_tiles)) + 1):
            if num_tiles % i == 0:
                factors.append((i, num_tiles // i))
        
        best_pair = factors[-1]
        return (best_pair[1], best_pair[0]) # Return as (cols, rows) for wider layout

    def close(self):
        pygame.quit()
        
    def render(self):
        # This method is not used by the agent, but can be useful for human viewing
        # We will use the _get_observation method's rendering logic
        self._update_and_draw_frame()
        # For direct display, we need to transpose back if needed, but here screen is correct
        return pygame.surfarray.array3d(self.screen)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed with display support.
    # To run, you might need to comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Glyph Decipher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # The observation is (H, W, C), which pygame can't blit directly.
        # We need to get the surface from the env *before* it's converted.
        # So we'll use the internal screen buffer for display.
        surf_to_show = pygame.transform.rotate(env.screen, -90)
        surf_to_show = pygame.transform.flip(surf_to_show, True, False)
        screen.blit(surf_to_show, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)

    env.close()