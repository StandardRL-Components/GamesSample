
# Generated: 2025-08-27T13:07:19.812551
# Source Brief: brief_00264.md
# Brief Index: 264

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A fast-paced, procedurally generated grid-based memory matching game where
    strategic risk-taking is rewarded. The goal is to find all matching pairs
    of cards on the grid.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: Use arrow keys to move the cursor. Press space to flip a card."

    # User-facing description of the game
    game_description = "A minimalist memory matching game. Find all the pairs to win. Score bonus points for quick consecutive matches."

    # The game state is static until a user submits an action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 6, 4
        self.CARD_COUNT = self.GRID_COLS * self.GRID_ROWS
        self.PAIR_COUNT = self.CARD_COUNT // 2
        self.MARGIN = 15
        self.GUTTER = 10
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_CARD_BACK = (52, 73, 94)
        self.COLOR_CARD_FACE = (236, 240, 241)
        self.COLOR_CURSOR = (241, 196, 15)
        self.COLOR_MATCHED_BG = (46, 204, 113)
        self.COLOR_MISMATCHED_BG = (231, 76, 60)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHAPE = (0, 0, 0)
        self.COLOR_SHAPE_INVERT = (255, 255, 255)

        # Calculate card dimensions
        grid_width = self.WIDTH - 2 * self.MARGIN
        grid_height = self.HEIGHT - 2 * self.MARGIN - 50  # Reserve 50px for UI
        self.CARD_WIDTH = (grid_width - (self.GRID_COLS - 1) * self.GUTTER) / self.GRID_COLS
        self.CARD_HEIGHT = (grid_height - (self.GRID_ROWS - 1) * self.GUTTER) / self.GRID_ROWS

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)

        # Initialize state variables
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.revealed = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=bool)
        self.cursor_pos = [0, 0]
        self.selected_cards = []
        self.steps = 0
        self.score = 0
        self.last_match_step = -100
        self.last_space_held = False
        self.is_mismatch_state = False
        self.rng = None

        self.reset()
        # self.validate_implementation() # For development; comment out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.last_match_step = -100
        self.last_space_held = False
        self.is_mismatch_state = False

        card_values = list(range(self.PAIR_COUNT)) * 2
        self.rng.shuffle(card_values)
        self.grid = np.array(card_values).reshape((self.GRID_ROWS, self.GRID_COLS))

        self.revealed = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=bool)
        self.cursor_pos = [0, 0]
        self.selected_cards = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        # On any new action, clear a mismatch from the previous turn
        if self.is_mismatch_state:
            self.selected_cards.clear()
            self.is_mismatch_state = False

        # Unpack actions
        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = bool(space_held)

        # Handle movement
        if movement != 0:
            if movement == 1:  # Up
                self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_ROWS
            elif movement == 2:  # Down
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
            elif movement == 3:  # Left
                self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_COLS
            elif movement == 4:  # Right
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

        # Handle selection
        if space_pressed and len(self.selected_cards) < 2:
            r, c = self.cursor_pos
            card_tuple = (r, c)
            if not self.revealed[r, c] and card_tuple not in self.selected_cards:
                # SFX: Card select sound
                self.selected_cards.append(card_tuple)

                if len(self.selected_cards) == 2:
                    r1, c1 = self.selected_cards[0]
                    r2, c2 = self.selected_cards[1]
                    
                    if self.grid[r1, c1] == self.grid[r2, c2]:
                        # Match found
                        # SFX: Match success sound
                        self.revealed[r1, c1] = True
                        self.revealed[r2, c2] = True
                        self.selected_cards.clear()

                        match_reward = 1.0
                        if self.steps - self.last_match_step <= 3:
                             # SFX: Combo bonus sound
                            match_reward += 5.0
                        reward += match_reward
                        self.last_match_step = self.steps
                    else:
                        # Mismatch
                        # SFX: Mismatch fail sound
                        reward -= 0.1
                        self.is_mismatch_state = True

        self.score += reward
        self.steps += 1

        # Check for termination conditions
        if np.all(self.revealed):
            # SFX: Game win fanfare
            reward += 50.0
            self.score += 50.0 # Add final bonus to score as well
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card_rect = self._get_card_rect(r, c)
                is_selected = (r, c) in self.selected_cards
                is_revealed = self.revealed[r, c]
                is_cursor_on = (r == self.cursor_pos[0] and c == self.cursor_pos[1])

                bg_color = self.COLOR_CARD_BACK
                if is_revealed:
                    bg_color = self.COLOR_MATCHED_BG
                elif is_selected:
                    bg_color = self.COLOR_MISMATCHED_BG if self.is_mismatch_state else self.COLOR_CARD_FACE
                
                pygame.draw.rect(self.screen, bg_color, card_rect, border_radius=5)
                
                if is_revealed or is_selected:
                    shape_color = self.COLOR_SHAPE_INVERT if self.is_mismatch_state else self.COLOR_SHAPE
                    self._draw_card_symbol(self.grid[r, c], card_rect, shape_color)

                if is_cursor_on:
                    pygame.draw.rect(self.screen, self.COLOR_CURSOR, card_rect, 4, border_radius=5)

    def _draw_card_symbol(self, value, rect, color):
        center_x, center_y = rect.center
        size = int(min(rect.width, rect.height) * 0.3)
        
        # Use geometric shapes for card faces
        shape_map = {
            0: lambda: self._draw_polygon(6, size, (center_x, center_y), color), # Hexagon
            1: lambda: self._draw_polygon(5, size, (center_x, center_y), color), # Pentagon
            2: lambda: self._draw_polygon(3, size, (center_x, center_y), color, math.pi/2), # Triangle
            3: lambda: self._draw_polygon(4, size, (center_x, center_y), color, math.pi/4), # Diamond
            4: lambda: self._draw_star(color, (center_x, center_y), size),
            5: lambda: self._draw_cross(color, (center_x, center_y), size),
            6: lambda: pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size, color),
            7: lambda: pygame.draw.rect(self.screen, color, (center_x - size, center_y - size, size * 2, size * 2)),
            8: lambda: self._draw_heart(color, (center_x, center_y), size),
            9: lambda: self._draw_club(color, (center_x, center_y), size),
            10: lambda: self._draw_spade(color, (center_x, center_y), size),
            11: lambda: self._draw_moon(color, (center_x, center_y), size),
        }
        if value in shape_map:
            shape_map[value]()

    def _draw_polygon(self, num_sides, size, center, color, rotation=0):
        points = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides + rotation
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_star(self, color, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36)
            radius = size if i % 2 == 0 else size * 0.4
            x = center[0] + radius * math.sin(angle)
            y = center[1] - radius * math.cos(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cross(self, color, center, size):
        w = int(size * 0.3)
        pygame.draw.rect(self.screen, color, (center[0] - size, center[1] - w, size * 2, w * 2))
        pygame.draw.rect(self.screen, color, (center[0] - w, center[1] - size, w * 2, size * 2))

    def _draw_heart(self, color, center, size):
        points = []
        for i in range(101):
            t = 2 * math.pi * i / 100
            x = center[0] + size * (16 * math.sin(t)**3) / 17
            y = center[1] - size * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 17
            points.append((int(x), int(y)))
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_club(self, color, center, size):
        x, y = center
        s = size * 0.5
        pygame.gfxdraw.filled_circle(self.screen, int(x-s), int(y), int(s), color)
        pygame.gfxdraw.filled_circle(self.screen, int(x+s), int(y), int(s), color)
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y-s), int(s), color)
        self._draw_polygon(3, s, (x,y+s*0.4), color, math.pi/2)

    def _draw_spade(self, color, center, size):
        x, y = center
        s = size
        self._draw_heart(color, (x, y - s*0.2), s*0.8)
        # Invert heart
        points = []
        for i in range(101):
            t = 2 * math.pi * i / 100
            px = x + s * (16 * math.sin(t)**3) / 17
            py = y - s * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 17
            points.append((int(px), int(py)))
        pygame.gfxdraw.filled_polygon(self.screen, [(p[0], 2*y-p[1]) for p in points], color)
        self._draw_polygon(3, s*0.7, (x, y+s*0.6), color, -math.pi/2)

    def _draw_moon(self, color, center, size):
        x, y = center
        s = size
        pygame.gfxdraw.filled_circle(self.screen, x, y, s, color)
        pygame.gfxdraw.filled_circle(self.screen, int(x+s*0.4), y, int(s*0.9), self.COLOR_CARD_FACE)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_ui.render(f"Moves: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_card_rect(self, row, col):
        top_offset = self.MARGIN + 50
        x = self.MARGIN + col * (self.CARD_WIDTH + self.GUTTER)
        y = top_offset + row * (self.CARD_HEIGHT + self.GUTTER)
        return pygame.Rect(x, y, self.CARD_WIDTH, self.CARD_HEIGHT)

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

# This block allows the game to be run directly for testing.
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Memory Match")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0])

    print("--- Playing Memory Match ---")
    print(env.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Movement is continuous while held
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0

        # Space and Shift are binary held/not-held
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.1f}, Score: {info['score']:.1f}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit FPS for human playability

    print("Game Over!")
    print(f"Final Score: {info['score']:.1f}, Total Steps: {info['steps']}")
    env.close()