
# Generated: 2025-08-27T18:14:30.624147
# Source Brief: brief_01770.md
# Brief Index: 1770

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A memory matching game environment for Gymnasium.

    The player must find all pairs of matching symbols on a grid before the
    timer runs out. The game features a clean visual style with particle effects
    and smooth animations, designed for a high-quality user experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to flip a card."
    )

    # User-facing description of the game
    game_description = (
        "Find all the matching pairs of symbols on the grid before time runs out. "
        "A fast-paced memory challenge!"
    )

    # Frames auto-advance for the real-time timer
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    GRID_ROWS = 4
    GRID_COLS = 6
    NUM_PAIRS = (GRID_ROWS * GRID_COLS) // 2
    MISMATCH_DELAY_FRAMES = 20

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_CARD_BACK = (70, 80, 100)
    COLOR_CARD_REVEALED = (180, 190, 210)
    COLOR_CARD_MATCHED_BG = (50, 60, 70, 150)
    COLOR_CARD_MATCHED_BORDER = (60, 70, 80, 150)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_MATCH = (0, 255, 120)
    COLOR_MISMATCH = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SYMBOL = (20, 30, 40)

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 60, bold=True)

        self._init_symbols()
        self.reset()
        
        self.validate_implementation()

    def _init_symbols(self):
        """Creates a list of lambda functions that draw different geometric symbols."""
        self.symbols = [
            # 0: Circle
            lambda s, p, sz, c: pygame.gfxdraw.aacircle(s, int(p[0]), int(p[1]), int(sz * 0.4), c) or pygame.gfxdraw.filled_circle(s, int(p[0]), int(p[1]), int(sz * 0.4), c),
            # 1: Square
            lambda s, p, sz, c: pygame.draw.rect(s, c, (p[0] - sz*0.4, p[1] - sz*0.4, sz*0.8, sz*0.8)),
            # 2: Triangle
            lambda s, p, sz, c: pygame.gfxdraw.aapolygon(s, [(p[0], p[1] - sz*0.45), (p[0] - sz*0.5, p[1] + sz*0.35), (p[0] + sz*0.5, p[1] + sz*0.35)], c) or pygame.gfxdraw.filled_polygon(s, [(p[0], p[1] - sz*0.45), (p[0] - sz*0.5, p[1] + sz*0.35), (p[0] + sz*0.5, p[1] + sz*0.35)], c),
            # 3: Diamond
            lambda s, p, sz, c: pygame.gfxdraw.aapolygon(s, [(p[0], p[1] - sz*0.5), (p[0] - sz*0.5, p[1]), (p[0], p[1] + sz*0.5), (p[0] + sz*0.5, p[1])], c) or pygame.gfxdraw.filled_polygon(s, [(p[0], p[1] - sz*0.5), (p[0] - sz*0.5, p[1]), (p[0], p[1] + sz*0.5), (p[0] + sz*0.5, p[1])], c),
            # 4: X
            lambda s, p, sz, c: pygame.draw.line(s, c, (p[0] - sz*0.4, p[1] - sz*0.4), (p[0] + sz*0.4, p[1] + sz*0.4), 5) or pygame.draw.line(s, c, (p[0] - sz*0.4, p[1] + sz*0.4), (p[0] + sz*0.4, p[1] - sz*0.4), 5),
            # 5: Plus
            lambda s, p, sz, c: pygame.draw.line(s, c, (p[0], p[1] - sz*0.4), (p[0], p[1] + sz*0.4), 5) or pygame.draw.line(s, c, (p[0] - sz*0.4, p[1]), (p[0] + sz*0.4, p[1]), 5),
            # 6: Star
            lambda s, p, sz, c: self._draw_star(s, p, sz*0.55, c),
            # 7: Hexagon
            lambda s, p, sz, c: self._draw_ngon(s, 6, p, sz*0.5, c),
            # 8: Pentagon
            lambda s, p, sz, c: self._draw_ngon(s, 5, p, sz*0.5, c),
            # 9: Ring
            lambda s, p, sz, c: pygame.draw.circle(s, c, p, sz*0.4, 5),
            # 10: Donut
            lambda s, p, sz, c: pygame.draw.circle(s, c, p, sz*0.4, 12),
            # 11: Hash
            lambda s, p, sz, c: pygame.draw.line(s, c, (p[0]-sz*0.4, p[1]-sz*0.2), (p[0]+sz*0.4, p[1]-sz*0.2), 5) or pygame.draw.line(s, c, (p[0]-sz*0.4, p[1]+sz*0.2), (p[0]+sz*0.4, p[1]+sz*0.2), 5) or pygame.draw.line(s, c, (p[0]-sz*0.2, p[1]-sz*0.4), (p[0]-sz*0.2, p[1]+sz*0.4), 5) or pygame.draw.line(s, c, (p[0]+sz*0.2, p[1]-sz*0.4), (p[0]+sz*0.2, p[1]+sz*0.4), 5),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self._setup_grid()
        self.cursor_pos = [0, 0]
        self.prev_action = np.array([0, 0, 0])
        self.first_selection_idx = None
        self.mismatch_timer = 0
        self.mismatched_pair_indices = []
        self.particles = []
        return self._get_observation(), self._get_info()

    def _setup_grid(self):
        symbol_indices = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(symbol_indices)
        self.cards = [
            {
                "symbol_id": symbol_indices[i],
                "state": "hidden",  # hidden, revealed, matched
                "first_revealed": False, # for reward
            }
            for i in range(self.GRID_ROWS * self.GRID_COLS)
        ]

    def step(self, action):
        reward = 0.0
        self.steps += 1
        self.time_left = max(0, self.time_left - 1)

        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                self.cards[self.mismatched_pair_indices[0]]["state"] = "hidden"
                self.cards[self.mismatched_pair_indices[1]]["state"] = "hidden"
                self.mismatched_pair_indices = []

        reward += self._handle_input(action)
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._check_win_condition():
                reward += 100.0  # Win bonus
            elif self.time_left <= 0:
                reward += -50.0  # Timeout penalty

        self.prev_action = action
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        prev_movement, prev_space_held, _ = self.prev_action[0], self.prev_action[1] == 1, self.prev_action[2] == 1

        # Treat change in movement as a single key press for grid navigation
        if movement != 0 and movement != prev_movement:
            if movement == 1: self.cursor_pos[0] -= 1  # Up
            if movement == 2: self.cursor_pos[0] += 1  # Down
            if movement == 3: self.cursor_pos[1] -= 1  # Left
            if movement == 4: self.cursor_pos[1] += 1  # Right
            self.cursor_pos[0] %= self.GRID_ROWS
            self.cursor_pos[1] %= self.GRID_COLS
            # sfx: cursor_move.wav

        # Treat rising edge of spacebar as a press
        space_pressed = space_held and not prev_space_held
        if space_pressed and self.mismatch_timer == 0 and not self.game_over:
            return self._handle_card_selection()
        
        return 0.0

    def _handle_card_selection(self):
        selected_idx = self.cursor_pos[0] * self.GRID_COLS + self.cursor_pos[1]
        card = self.cards[selected_idx]
        reward = 0.0

        if card["state"] == "hidden":
            # sfx: card_flip.wav
            card["state"] = "revealed"
            if not card["first_revealed"]:
                reward += 0.1
                card["first_revealed"] = True

            if self.first_selection_idx is None:
                self.first_selection_idx = selected_idx
            else:
                if self.first_selection_idx == selected_idx: # Clicked same card twice
                    return 0.0
                    
                first_card = self.cards[self.first_selection_idx]
                if first_card["symbol_id"] == card["symbol_id"]:
                    # Match
                    first_card["state"] = "matched"
                    card["state"] = "matched"
                    self.score += 1
                    reward += 10.0
                    self._create_match_particles(self._get_card_rect(*divmod(selected_idx, self.GRID_COLS)).center)
                    self._create_match_particles(self._get_card_rect(*divmod(self.first_selection_idx, self.GRID_COLS)).center)
                    # sfx: match_success.wav
                else:
                    # Mismatch
                    reward -= 1.0
                    self.mismatch_timer = self.MISMATCH_DELAY_FRAMES
                    self.mismatched_pair_indices = [self.first_selection_idx, selected_idx]
                    # sfx: mismatch_fail.wav
                self.first_selection_idx = None
        return reward

    def _check_win_condition(self):
        return all(c["state"] == "matched" for c in self.cards)

    def _check_termination(self):
        if self._check_win_condition():
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_cards()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left}
        
    def _get_card_rect(self, row, col):
        grid_rect = pygame.Rect(40, 60, 560, 320)
        card_w = grid_rect.width / self.GRID_COLS * 0.85
        card_h = grid_rect.height / self.GRID_ROWS * 0.85
        gap_x = (grid_rect.width - self.GRID_COLS * card_w) / (self.GRID_COLS + 1)
        gap_y = (grid_rect.height - self.GRID_ROWS * card_h) / (self.GRID_ROWS + 1)
        
        x = grid_rect.left + gap_x + col * (card_w + gap_x)
        y = grid_rect.top + gap_y + row * (card_h + gap_y)
        return pygame.Rect(x, y, card_w, card_h)

    def _render_grid_and_cards(self):
        grid_rect = pygame.Rect(40, 60, 560, 320)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                idx = r * self.GRID_COLS + c
                card = self.cards[idx]
                rect = self._get_card_rect(r, c)
                
                if card["state"] == "matched":
                    pygame.gfxdraw.box(self.screen, rect, self.COLOR_CARD_MATCHED_BG)
                    pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_CARD_MATCHED_BORDER)
                else:
                    color = self.COLOR_CARD_BACK
                    if card["state"] == "revealed":
                        color = self.COLOR_CARD_REVEALED
                        if self.mismatch_timer > 0 and idx in self.mismatched_pair_indices:
                            # Flash red on mismatch
                            t = self.mismatch_timer / self.MISMATCH_DELAY_FRAMES
                            color = self._interpolate_color(self.COLOR_MISMATCH, self.COLOR_CARD_REVEALED, 1 - t)
                    
                    pygame.draw.rect(self.screen, color, rect, border_radius=8)
                    if card["state"] == "revealed":
                        self.symbols[card["symbol_id"]](self.screen, rect.center, rect.width, self.COLOR_SYMBOL)

    def _render_cursor(self):
        if not self.game_over:
            rect = self._get_card_rect(self.cursor_pos[0], self.cursor_pos[1])
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect.inflate(8, 8), width=4, border_radius=12)

    def _render_ui(self):
        # Timer Bar
        timer_rect = pygame.Rect(40, 20, self.SCREEN_WIDTH - 80, 25)
        timer_ratio = max(0, self.time_left / (self.TIME_LIMIT_SECONDS * self.FPS))
        current_width = int(timer_rect.width * timer_ratio)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, timer_rect, border_radius=5)
        if current_width > 0:
            color = self._interpolate_color(self.COLOR_MISMATCH, self.COLOR_CURSOR, timer_ratio)
            pygame.draw.rect(self.screen, color, (timer_rect.x, timer_rect.y, current_width, timer_rect.height), border_radius=5)
        
        # Score Text
        score_text = f"PAIRS: {self.score} / {self.NUM_PAIRS}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (timer_rect.right - text_surf.get_width() - 10, timer_rect.centery - text_surf.get_height()//2))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        win = self._check_win_condition()
        text = "YOU WIN!" if win else "TIME'S UP!"
        color = self.COLOR_MATCH if win else self.COLOR_MISMATCH
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _create_match_particles(self, pos):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            life = random.randint(self.FPS // 2, self.FPS)
            size = random.uniform(2, 6)
            color = random.choice([self.COLOR_MATCH, self.COLOR_CURSOR, (200, 255, 220)])
            self.particles.append([list(pos), [math.cos(angle) * speed, math.sin(angle) * speed], life, size, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1  # Gravity
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos, _, life, size, color = p
            alpha = int(255 * (life / self.FPS))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (size, size), size)
            self.screen.blit(s, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_ngon(self, surface, n, center, radius, color):
        points = []
        for i in range(n):
            angle = i * (2 * math.pi / n) - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_star(self, surface, center, radius, color):
        points = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.45
            angle = i * (2 * math.pi / 10) - math.pi / 2
            points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _interpolate_color(self, color1, color2, factor):
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Memory Match")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Event handling ---
        action = np.array([0, 0, 0]) # Default no-op
        keys = pygame.key.get_pressed()

        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation ---
        # The observation is a numpy array, convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

        clock.tick(GameEnv.FPS)

    env.close()