
# Generated: 2025-08-27T15:09:55.289503
# Source Brief: brief_00908.md
# Brief Index: 908

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, rng):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.x) - radius, int(self.y) - radius))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move the cursor. Press Space to flip a card."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced memory match game. Find all pairs before time runs out or you make too many mistakes."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 4, 8
        self.NUM_PAIRS = (self.GRID_ROWS * self.GRID_COLS) // 2
        self.MAX_MISTAKES = 3
        self.TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.MAX_STEPS = self.FPS * self.TIME_LIMIT_SECONDS + 10 # A bit over the time limit

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CARD_BACK = (60, 80, 100)
        self.COLOR_CARD_FRONT = (210, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_MATCH = (0, 255, 120)
        self.COLOR_MISMATCH = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TIMER = (255, 255, 0)
        self.COLOR_TIMER_LOW = (255, 0, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Card layout
        self.card_padding = 8
        card_area_w = self.WIDTH - self.card_padding * (self.GRID_COLS + 1)
        card_area_h = self.HEIGHT - 100 - self.card_padding * (self.GRID_ROWS + 1)
        self.card_w = card_area_w / self.GRID_COLS
        self.card_h = card_area_h / self.GRID_ROWS
        self.grid_offset_x = self.card_padding
        self.grid_offset_y = 60

        # Initialize state variables
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.timer = float(self.TIME_LIMIT_SECONDS)
        self.mistakes = 0
        self.matched_pairs = 0

        self.cursor_pos = [0, 0] # [row, col]
        self.space_was_pressed = False
        
        self.first_selection = None
        self.mismatched_pair = []
        self.mismatch_timer = 0

        self.particles = []
        self._create_grid()

        return self._get_observation(), self._get_info()

    def _create_grid(self):
        symbols = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(symbols)
        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                symbol = symbols.pop()
                card = {
                    "symbol": symbol,
                    "state": "hidden", # hidden, revealed, matched
                    "flip_progress": 0.0, # 0 = hidden, 1 = revealed
                    "target_flip": 0.0
                }
                row.append(card)
            self.grid.append(row)

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update timers
        self.timer = max(0, self.timer - 1.0 / self.FPS)
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                r1, c1 = self.mismatched_pair[0]
                r2, c2 = self.mismatched_pair[1]
                self.grid[r1][c1]["target_flip"] = 0.0 # Flip back
                self.grid[r2][c2]["target_flip"] = 0.0 # Flip back
                self.mismatched_pair = []

        # Update card flip animations
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                if card["flip_progress"] != card["target_flip"]:
                    diff = card["target_flip"] - card["flip_progress"]
                    card["flip_progress"] += diff * 0.2 # Smooth interpolation

        # Handle input
        self._handle_movement(movement)
        reward += self._handle_selection(space_held)
        self.space_was_pressed = space_held

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        terminal_reward = 0
        
        if self.matched_pairs == self.NUM_PAIRS:
            terminated = True
            self.win = True
            terminal_reward = 100.0
        elif self.mistakes >= self.MAX_MISTAKES:
            terminated = True
            terminal_reward = -100.0
        elif self.timer <= 0:
            terminated = True
            terminal_reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        reward += terminal_reward
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

    def _handle_selection(self, space_held):
        reward = 0
        is_press = space_held and not self.space_was_pressed
        if not is_press:
            return reward

        r, c = self.cursor_pos
        card = self.grid[r][c]

        if card["state"] != "hidden" or self.mismatch_timer > 0:
            reward -= 0.1 # Penalty for selecting invalid card
            return reward

        reward += 0.1 # Reward for valid selection
        card["state"] = "revealed"
        card["target_flip"] = 1.0

        if self.first_selection is None:
            self.first_selection = (r, c)
        else:
            r1, c1 = self.first_selection
            card1 = self.grid[r1][c1]
            card2 = card

            if card1["symbol"] == card2["symbol"]:
                # Match
                card1["state"] = "matched"
                card2["state"] = "matched"
                self.matched_pairs += 1
                reward += 10.0
                pos1 = self._get_card_center(r1, c1)
                pos2 = self._get_card_center(r, c)
                self._create_particles(pos1, self.COLOR_MATCH, 30)
                self._create_particles(pos2, self.COLOR_MATCH, 30)
                self.first_selection = None
            else:
                # Mismatch
                card1["state"] = "hidden"
                card2["state"] = "hidden"
                self.mistakes += 1
                reward -= 5.0
                self.mismatched_pair = [(r1, c1), (r, c)]
                self.mismatch_timer = self.FPS # 1 second
                pos1 = self._get_card_center(r1, c1)
                pos2 = self._get_card_center(r, c)
                self._create_particles(pos1, self.COLOR_MISMATCH, 20)
                self._create_particles(pos2, self.COLOR_MISMATCH, 20)
                self.first_selection = None
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_cards()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mistakes": self.mistakes,
            "matched_pairs": self.matched_pairs,
            "time_left": self.timer,
        }

    def _render_grid_bg(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = self._get_card_rect(r, c)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=5)

    def _render_cards(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                if card["state"] == "matched":
                    continue
                
                rect = self._get_card_rect(r, c)
                flip_scale = abs(math.cos(card["flip_progress"] * math.pi / 2))
                
                scaled_width = int(rect.width * flip_scale)
                if scaled_width <= 0: continue
                
                display_rect = pygame.Rect(
                    rect.centerx - scaled_width // 2,
                    rect.y,
                    scaled_width,
                    rect.height
                )

                is_showing_front = card["flip_progress"] > 0.5
                
                if is_showing_front:
                    pygame.draw.rect(self.screen, self.COLOR_CARD_FRONT, display_rect, border_radius=4)
                    symbol_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    self._render_card_symbol(symbol_surface, card["symbol"], symbol_surface.get_rect())
                    
                    scaled_symbol = pygame.transform.scale(symbol_surface, (scaled_width, rect.height))
                    self.screen.blit(scaled_symbol, display_rect.topleft)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, display_rect, border_radius=4)

    def _render_card_symbol(self, surface, symbol_id, rect):
        color = self.COLOR_BG
        center = rect.center
        size = min(rect.width, rect.height) * 0.35
        
        s = symbol_id % 16
        if s == 0: # Circle
            pygame.gfxdraw.aacircle(surface, center[0], center[1], int(size), color)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(size), color)
        elif s == 1: # Square
            pygame.draw.rect(surface, color, (center[0] - size, center[1] - size, size * 2, size * 2))
        elif s == 2: # Triangle
            points = [(center[0], center[1] - size), (center[0] - size, center[1] + size), (center[0] + size, center[1] + size)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif s == 3: # Diamond
            points = [(center[0], center[1] - size), (center[0] + size, center[1]), (center[0], center[1] + size), (center[0] - size, center[1])]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif s == 4: # X
            pygame.draw.line(surface, color, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), 5)
            pygame.draw.line(surface, color, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), 5)
        elif s == 5: # Plus
            pygame.draw.line(surface, color, (center[0], center[1] - size), (center[0], center[1] + size), 5)
            pygame.draw.line(surface, color, (center[0] - size, center[1]), (center[0] + size, center[1]), 5)
        elif s == 6: # Hexagon
            points = [(center[0] + size * math.cos(math.radians(a)), center[1] + size * math.sin(math.radians(a))) for a in range(0, 360, 60)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif s == 7: # Star
            points = []
            for i in range(10):
                r = size if i % 2 == 0 else size * 0.5
                angle = math.radians(i * 36 - 90)
                points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        else: # Fallback symbols
            p = [ ( (s>>i)&1 ) for i in range(4) ] # 4 bits from symbol
            pygame.draw.rect(surface, color, (center[0]-size*0.8, center[1]-size*0.8, size*0.7, size*0.7), 0 if p[0] else 3)
            pygame.draw.rect(surface, color, (center[0]+size*0.1, center[1]-size*0.8, size*0.7, size*0.7), 0 if p[1] else 3)
            pygame.draw.rect(surface, color, (center[0]-size*0.8, center[1]+size*0.1, size*0.7, size*0.7), 0 if p[2] else 3)
            pygame.draw.rect(surface, color, (center[0]+size*0.1, center[1]+size*0.1, size*0.7, size*0.7), 0 if p[3] else 3)

    def _render_cursor(self):
        if self.game_over: return
        r, c = self.cursor_pos
        rect = self._get_card_rect(r, c).inflate(8, 8)
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        color = tuple(int(c1 + (c2 - c1) * pulse) for c1, c2 in zip(self.COLOR_CURSOR, (255, 255, 150)))
        
        pygame.draw.rect(self.screen, color, rect, 3, border_radius=8)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {self.timer:.1f}"
        timer_color = self.COLOR_TIMER if self.timer > 10 else self.COLOR_TIMER_LOW
        text_surf = self.font_small.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (20, 20))

        # Mistakes
        mistakes_text = "MISTAKES:"
        text_surf = self.font_small.render(mistakes_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - 200, 20))
        for i in range(self.MAX_MISTAKES):
            color = self.COLOR_MISMATCH if i < self.mistakes else self.COLOR_GRID
            pygame.draw.circle(self.screen, color, (self.WIDTH - 100 + i * 25, 28), 8)

        # Matched Pairs
        matched_text = f"PAIRS: {self.matched_pairs}/{self.NUM_PAIRS}"
        text_surf = self.font_small.render(matched_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 20))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_MATCH if self.win else self.COLOR_MISMATCH
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)
        
    def _get_card_rect(self, r, c):
        x = self.grid_offset_x + c * (self.card_w + self.card_padding) + self.card_padding
        y = self.grid_offset_y + r * (self.card_h + self.card_padding) + self.card_padding
        return pygame.Rect(x, y, self.card_w, self.card_h)

    def _get_card_center(self, r, c):
        rect = self._get_card_rect(r, c)
        return rect.center

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, self.FPS // 2, self.np_random))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Memory Match")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Main game loop
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Optional: auto-reset after a delay
            # pygame.time.wait(2000)
            # obs, info = env.reset()
            # total_reward = 0
            
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose the observation back to the format pygame expects
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(env.FPS)
        
    pygame.quit()