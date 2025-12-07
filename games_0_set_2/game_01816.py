# Generated: 2025-08-27T18:24:28.114115
# Source Brief: brief_01816.md
# Brief Index: 1816


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
        "Controls: Use arrow keys to move the cursor. Press Space to flip a card."
    )

    game_description = (
        "A classic memory game. Find all matching pairs of cards before the time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIMER_DURATION = 60.0  # 60 seconds
        self.MAX_STEPS = int(self.TIMER_DURATION * self.FPS)

        # Grid and Card Dimensions
        self.GRID_ROWS, self.GRID_COLS = 4, 6
        self.NUM_PAIRS = (self.GRID_ROWS * self.GRID_COLS) // 2
        self.CARD_W, self.CARD_H = 80, 70
        self.SPACING = 12
        grid_total_w = self.GRID_COLS * self.CARD_W + (self.GRID_COLS - 1) * self.SPACING
        grid_total_h = self.GRID_ROWS * self.CARD_H + (self.GRID_ROWS - 1) * self.SPACING
        self.GRID_X_START = (self.WIDTH - grid_total_w) // 2
        self.GRID_Y_START = 80

        # Colors
        self.COLOR_BG = (25, 35, 55)
        self.COLOR_CARD_BACK = (70, 80, 100)
        self.COLOR_CARD_FACE = (210, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_MATCH = (40, 220, 110)
        self.COLOR_MISMATCH = (255, 70, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TIMER_WARN = (255, 150, 0)
        self.COLOR_TIMER_CRIT = (255, 50, 50)
        self.SYMBOL_COLORS = [
            (231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113),
            (26, 188, 156), (52, 152, 219), (155, 89, 182), (255, 118, 117),
            (9, 132, 227), (0, 184, 148), (253, 203, 110), (108, 92, 231)
        ]

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.cursor_pos = [0, 0]
        self.grid = []
        self.first_selection = None
        self.mismatch_timer = 0
        self.mismatched_pair = []
        self.match_animation_timer = 0
        self.just_matched_pair = []
        self.matched_pairs_count = 0
        self.prev_space_held = False

        self.np_random = None

    def _get_symbol_draw_funcs(self):
        # List of functions to draw unique symbols
        return [
            self._draw_circle, self._draw_square, self._draw_triangle, self._draw_diamond,
            self._draw_star, self._draw_cross, self._draw_hexagon, self._draw_pentagon,
            self._draw_pacman, self._draw_heart, self._draw_club, self._draw_spade
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIMER_DURATION
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.mismatch_timer = 0
        self.mismatched_pair = []
        self.match_animation_timer = 0
        self.just_matched_pair = []
        self.matched_pairs_count = 0
        self.prev_space_held = False

        # Create and shuffle cards
        card_values = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(card_values)

        symbol_funcs = self._get_symbol_draw_funcs()

        self.grid = np.array([
            {
                'value': card_values[r * self.GRID_COLS + c],
                'state': 'down',  # 'down', 'up', 'matched'
                'symbol_func': symbol_funcs[card_values[r * self.GRID_COLS + c]],
                'color': self.SYMBOL_COLORS[card_values[r * self.GRID_COLS + c]]
            }
            for r in range(self.GRID_ROWS)
            for c in range(self.GRID_COLS)
        ]).reshape((self.GRID_ROWS, self.GRID_COLS))

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Update Animation Timers ---
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                r1, c1 = self.mismatched_pair[0]
                r2, c2 = self.mismatched_pair[1]
                self.grid[r1, c1]['state'] = 'down'
                self.grid[r2, c2]['state'] = 'down'
                self.mismatched_pair = []

        if self.match_animation_timer > 0:
            self.match_animation_timer -= 1
            if self.match_animation_timer == 0:
                self.just_matched_pair = []

        # --- Handle Input and Game Logic ---
        self._handle_movement(movement)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._handle_flip()
        self.prev_space_held = space_held

        # --- Update Game State ---
        self.timer = max(0, self.timer - 1.0 / self.FPS)
        self.steps += 1

        # --- Check for Termination ---
        terminated = False
        win_condition = self.matched_pairs_count == self.NUM_PAIRS
        lose_condition = self.timer <= 0 or self.steps >= self.MAX_STEPS

        if win_condition or lose_condition:
            terminated = True
            if win_condition and not self.game_over:
                # Add win bonus only once
                win_bonus = 50.0
                reward += win_bonus
                self.score += win_bonus
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_ROWS
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_COLS
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

    def _handle_flip(self):
        # Can't flip while a mismatch is being shown
        if self.mismatch_timer > 0:
            return 0.0

        r, c = self.cursor_pos
        card = self.grid[r, c]

        # Can only flip face-down cards
        if card['state'] != 'down':
            return 0.0

        card['state'] = 'up'
        # SFX: card_flip.wav

        if self.first_selection is None:
            self.first_selection = (r, c)
        else:
            r1, c1 = self.first_selection
            card1 = self.grid[r1, c1]
            # Check for match
            if card1['value'] == card['value']:
                # --- MATCH ---
                card['state'] = 'matched'
                card1['state'] = 'matched'
                self.matched_pairs_count += 1
                self.score += 1.0
                self.first_selection = None
                self.just_matched_pair = [(r1, c1), (r, c)]
                self.match_animation_timer = int(self.FPS * 0.5)  # 0.5s glow
                # SFX: match_success.wav
                return 1.0
            else:
                # --- MISMATCH ---
                self.score -= 0.1
                self.mismatched_pair = [(r1, c1), (r, c)]
                self.mismatch_timer = int(self.FPS * 0.75)  # 0.75s view time
                self.first_selection = None
                # SFX: mismatch_fail.wav
                return -0.1
        return 0.0

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
            "timer": self.timer,
            "matched_pairs": self.matched_pairs_count,
        }

    def _render_game(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r, c]
                card_x = self.GRID_X_START + c * (self.CARD_W + self.SPACING)
                card_y = self.GRID_Y_START + r * (self.CARD_H + self.SPACING)
                rect = pygame.Rect(card_x, card_y, self.CARD_W, self.CARD_H)

                # Draw glow effects first
                is_mismatch = (r, c) in self.mismatched_pair
                is_match = (r, c) in self.just_matched_pair
                if is_mismatch and self.mismatch_timer > 0:
                    self._draw_glow(rect, self.COLOR_MISMATCH, self.mismatch_timer)
                if is_match and self.match_animation_timer > 0:
                    self._draw_glow(rect, self.COLOR_MATCH, self.match_animation_timer)

                # Draw card base
                if card['state'] == 'down':
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, rect, border_radius=5)
                else:  # 'up' or 'matched'
                    face_color = self.COLOR_CARD_FACE
                    if card['state'] == 'matched':
                        # Fade matched cards slightly
                        face_color = tuple(int(x * 0.8) for x in face_color)
                    pygame.draw.rect(self.screen, face_color, rect, border_radius=5)
                    # Draw symbol
                    symbol_rect = rect.inflate(-20, -20)
                    card['symbol_func'](self.screen, symbol_rect, card['color'])

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.GRID_X_START + cursor_c * (self.CARD_W + self.SPACING)
        cursor_y = self.GRID_Y_START + cursor_r * (self.CARD_H + self.SPACING)
        cursor_rect = pygame.Rect(cursor_x - 4, cursor_y - 4, self.CARD_W + 8, self.CARD_H + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)

    def _draw_glow(self, rect, color, timer):
        # Pulsing glow effect
        alpha = 96 + math.sin(self.steps * 0.5) * 32
        radius = rect.width // 2 + 10
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
        self.screen.blit(s, (rect.centerx - radius, rect.centery - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{int(round(self.score))}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (30, 20))
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (30, 50))

        # Matched Pairs
        pairs_text = self.font_large.render(f"{self.matched_pairs_count}/{self.NUM_PAIRS}", True, self.COLOR_TEXT)
        pairs_rect = pairs_text.get_rect(centerx=self.WIDTH // 2, y=20)
        self.screen.blit(pairs_text, pairs_rect)
        pairs_label = self.font_small.render("PAIRS", True, self.COLOR_TEXT)
        pairs_label_rect = pairs_label.get_rect(centerx=self.WIDTH // 2, y=50)
        self.screen.blit(pairs_label, pairs_label_rect)

        # Timer
        time_str = f"{self.timer:.1f}"
        timer_color = self.COLOR_TEXT
        if self.timer < 10:
            timer_color = self.COLOR_TIMER_CRIT
        elif self.timer < 20:
            timer_color = self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(time_str, True, timer_color)
        timer_rect = timer_text.get_rect(right=self.WIDTH - 30, y=20)
        self.screen.blit(timer_text, timer_rect)
        timer_label = self.font_small.render("TIME", True, self.COLOR_TEXT)
        timer_label_rect = timer_label.get_rect(right=self.WIDTH - 30, y=50)
        self.screen.blit(timer_label, timer_label_rect)

    # --- Symbol Drawing Functions ---
    def _draw_circle(self, s, r, c): pygame.gfxdraw.filled_circle(s, r.centerx, r.centery, min(r.width, r.height) // 2, c)
    def _draw_square(self, s, r, c): pygame.draw.rect(s, c, r)
    def _draw_triangle(self, s, r, c): pygame.draw.polygon(s, c, [(r.midtop), (r.bottomleft), (r.bottomright)])
    def _draw_diamond(self, s, r, c): pygame.draw.polygon(s, c, [r.midtop, r.midright, r.midbottom, r.midleft])
    def _draw_star(self, s, r, c):
        points = []
        outer_r = min(r.width, r.height) / 2
        inner_r = outer_r / 2.5
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            radius = outer_r if i % 2 == 0 else inner_r
            points.append((r.centerx + radius * math.cos(angle), r.centery + radius * math.sin(angle)))
        pygame.draw.polygon(s, c, points)

    def _draw_cross(self, s, r, c):
        pygame.draw.rect(s, c, (r.centerx - r.width // 8, r.top, r.width // 4, r.height))
        pygame.draw.rect(s, c, (r.left, r.centery - r.height // 8, r.width, r.height // 4))

    def _draw_hexagon(self, s, r, c):
        radius = min(r.width, r.height) / 2
        points = [(r.centerx + radius * math.cos(math.radians(60 * i)), r.centery + radius * math.sin(math.radians(60 * i))) for i in range(6)]
        pygame.draw.polygon(s, c, points)

    def _draw_pentagon(self, s, r, c):
        radius = min(r.width, r.height) / 2
        points = [(r.centerx + radius * math.cos(math.radians(72 * i - 90)), r.centery + radius * math.sin(math.radians(72 * i - 90))) for i in range(5)]
        pygame.draw.polygon(s, c, points)

    def _draw_pacman(self, s, r, c):
        radius = int(min(r.width, r.height) / 2)
        center = r.center
        
        # Angles for the arc (in degrees, clockwise from 3 o'clock)
        start_angle_deg = 45
        end_angle_deg = 315
        
        # The points for the polygon start with the center
        points = [center]
        
        # Generate points along the arc
        num_segments = 50 # Use a fixed number of segments for a smooth curve
        
        for i in range(num_segments + 1):
            # Interpolate the angle
            angle_deg = start_angle_deg + (end_angle_deg - start_angle_deg) * i / num_segments
            
            # Convert degrees to radians for math functions
            angle_rad = math.radians(angle_deg)
            
            # In Pygame, a positive y-axis is downwards.
            # To use standard math angles (CCW from +x), you'd do y = center_y - r*sin(angle).
            # To use clockwise angles from +x, you can do y = center_y + r*sin(angle).
            # This matches the expected behavior of gfxdraw's pie function.
            x = center[0] + radius * math.cos(angle_rad)
            y = center[1] + radius * math.sin(angle_rad)
            points.append((x, y))
            
        # Draw the filled polygon
        if len(points) > 2:
            pygame.draw.polygon(s, c, points)

    def _draw_heart(self, s, r, c):
        x, y, w, h = r
        w2, h2 = w / 2, h / 2
        pygame.draw.polygon(s, c, [(x + w2, y + h2 + h / 4), (x, y + h / 4), (x + w2, y + h * 0.75)])
        pygame.draw.polygon(s, c, [(x + w2, y + h2 + h / 4), (x + w, y + h / 4), (x + w2, y + h * 0.75)])
        pygame.gfxdraw.filled_circle(s, int(x + w / 4), int(y + h / 4), int(w / 4), c)
        pygame.gfxdraw.filled_circle(s, int(x + w * 3 / 4), int(y + h / 4), int(w / 4), c)

    def _draw_club(self, s, r, c):
        x, y, w, h = r
        pygame.gfxdraw.filled_circle(s, int(x + w / 4), int(y + h * 2 / 3), int(w / 4), c)
        pygame.gfxdraw.filled_circle(s, int(x + w * 3 / 4), int(y + h * 2 / 3), int(w / 4), c)
        pygame.gfxdraw.filled_circle(s, int(x + w / 2), int(y + h / 3), int(w / 4), c)
        pygame.draw.polygon(s, c, [(x + w / 2, y + h / 2), (x + w * 0.4, y + h), (x + w * 0.6, y + h)])

    def _draw_spade(self, s, r, c):
        x, y, w, h = r
        pygame.gfxdraw.filled_circle(s, int(x + w / 4), int(y + h / 2), int(w / 4), c)
        pygame.gfxdraw.filled_circle(s, int(x + w * 3 / 4), int(y + h / 2), int(w / 4), c)
        pygame.draw.polygon(s, c, [(x + w / 2, y), (x, y + h * 0.6), (x + w, y + h * 0.6)])
        pygame.draw.polygon(s, c, [(x + w / 2, y + h * 0.6), (x + w * 0.4, y + h), (x + w * 0.6, y + h)])

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # This method is for the developer to check their implementation
        # It is not part of the standard gym.Env API
        try:
            # Test action space
            assert isinstance(self.action_space, MultiDiscrete)
            assert self.action_space.nvec.tolist() == [5, 2, 2]

            # Test observation space
            assert isinstance(self.observation_space, Box)
            assert self.observation_space.shape == (self.HEIGHT, self.WIDTH, 3)
            assert self.observation_space.dtype == np.uint8

            # Test reset
            obs, info = self.reset(seed=123)
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
            assert obs.dtype == np.uint8
            assert isinstance(info, dict)

            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
            assert obs.dtype == np.uint8
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)

            print("✓ Implementation validated successfully")
        except AssertionError as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == "__main__":
    # --- Manual Play ---
    # This part is for human interaction and will not run in the evaluation
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    pygame.init() # Re-init with display
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    terminated = False

    # Pygame setup for display
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Memory Match")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0])

    print("\n" + "=" * 30)
    print("MANUAL PLAY")
    print(env.user_guide)
    print("=" * 30 + "\n")

    while not terminated:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()

        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Space
        if keys[pygame.K_SPACE]: action[1] = 1

        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Render to Screen ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()