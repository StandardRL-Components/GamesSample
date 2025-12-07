
# Generated: 2025-08-28T03:04:22.859747
# Source Brief: brief_01903.md
# Brief Index: 1903

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press space to flip a card. "
        "Match all pairs before the time runs out!"
    )

    game_description = (
        "A fast-paced memory game. Flip cards to reveal geometric patterns. "
        "Find all 16 matching pairs against the clock to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 4
    MAX_STEPS = 500
    MISMATCH_DELAY_STEPS = 3  # Steps to show a mismatch before flipping back

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_CARD_HIDDEN = (55, 63, 81)
    COLOR_CARD_REVEALED = (75, 85, 109)
    COLOR_CARD_MATCHED = (35, 40, 51) # Slightly darker than BG
    COLOR_CURSOR = (255, 215, 0)
    COLOR_MATCH_FLASH = (74, 222, 128, 150)
    COLOR_MISMATCH_FLASH = (239, 68, 68, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PROGRESS_BAR = (60, 160, 240)
    COLOR_PROGRESS_BAR_BG = (40, 45, 55)
    
    PATTERN_COLORS = [
        (255, 82, 82), (255, 179, 64), (255, 255, 102), (124, 252, 0),
        (0, 255, 255), (0, 127, 255), (139, 0, 255), (255, 0, 127)
    ]

    # --- Card States ---
    STATE_HIDDEN = 0
    STATE_REVEALED = 1
    STATE_MATCHED = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.second_selection = None
        self.mismatch_timer = 0
        self.last_space_state = 0
        self.steps = 0
        self.score = 0
        self.matched_pairs_count = 0
        self.game_over = False

        self.reset()
        
        # This can be commented out after validation, but is good for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.matched_pairs_count = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.second_selection = None
        self.mismatch_timer = 0
        self.last_space_state = 0
        
        # Generate and shuffle cards
        pattern_ids = list(range(16)) * 2
        self.np_random.shuffle(pattern_ids)
        
        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                pattern_id = pattern_ids.pop()
                card = {
                    "pattern": pattern_id,
                    "state": self.STATE_HIDDEN,
                    "pos": (c, r)
                }
                row.append(card)
            self.grid.append(row)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Handle Mismatch Timer ---
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                # Flip cards back to hidden
                c1 = self.grid[self.first_selection[1]][self.first_selection[0]]
                c2 = self.grid[self.second_selection[1]][self.second_selection[0]]
                c1["state"] = self.STATE_HIDDEN
                c2["state"] = self.STATE_HIDDEN
                self.first_selection = None
                self.second_selection = None

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Movement ---
        if self.mismatch_timer == 0: # Prevent movement during mismatch view
            if movement == 1:  # Up
                self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_ROWS
            elif movement == 2:  # Down
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
            elif movement == 3:  # Left
                self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_COLS
            elif movement == 4:  # Right
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS

        # --- Handle Card Selection (Space Press) ---
        is_space_press = space_held and not self.last_space_state
        if is_space_press and self.mismatch_timer == 0:
            card = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
            
            if card["state"] == self.STATE_HIDDEN:
                card["state"] = self.STATE_REVEALED
                reward += 0.1 # Reward for exploration
                # sound: flip_card.wav
                
                if self.first_selection is None:
                    self.first_selection = tuple(self.cursor_pos)
                else:
                    self.second_selection = tuple(self.cursor_pos)
                    
                    # --- Check for Match ---
                    card1_data = self.grid[self.first_selection[1]][self.first_selection[0]]
                    card2_data = self.grid[self.second_selection[1]][self.second_selection[0]]
                    
                    if card1_data["pattern"] == card2_data["pattern"]:
                        # Match found!
                        card1_data["state"] = self.STATE_MATCHED
                        card2_data["state"] = self.STATE_MATCHED
                        self.matched_pairs_count += 1
                        reward += 10
                        self.score += 10 # Add to persistent score
                        self.first_selection = None
                        self.second_selection = None
                        # sound: match.wav
                        
                        if self.matched_pairs_count == (self.GRID_COLS * self.GRID_ROWS) // 2:
                            reward += 100
                            self.score += 100
                            terminated = True
                            # sound: win_game.wav
                    else:
                        # Mismatch
                        self.mismatch_timer = self.MISMATCH_DELAY_STEPS
                        # sound: mismatch.wav
            else:
                # Penalty for selecting an already revealed/matched card
                reward -= 0.01

        self.last_space_state = space_held
        self.steps += 1
        
        # --- Check Termination Conditions ---
        if self.steps >= self.MAX_STEPS:
            terminated = True
            # sound: lose_game.wav

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "matched_pairs": self.matched_pairs_count,
        }

    def _render_game(self):
        padding = 20
        board_width = self.SCREEN_WIDTH - 2 * padding
        board_height = self.SCREEN_HEIGHT - 60 # Reserve space for UI
        
        card_gap = 8
        card_width = (board_width - (self.GRID_COLS - 1) * card_gap) / self.GRID_COLS
        card_height = (board_height - (self.GRID_ROWS - 1) * card_gap) / self.GRID_ROWS

        start_x = padding
        start_y = 10

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                card_rect = pygame.Rect(
                    int(start_x + c * (card_width + card_gap)),
                    int(start_y + r * (card_height + card_gap)),
                    int(card_width),
                    int(card_height)
                )

                # Draw card body
                color = self.COLOR_CARD_HIDDEN
                if card["state"] == self.STATE_REVEALED:
                    color = self.COLOR_CARD_REVEALED
                elif card["state"] == self.STATE_MATCHED:
                    color = self.COLOR_CARD_MATCHED
                
                pygame.draw.rect(self.screen, color, card_rect, border_radius=5)
                
                # Draw pattern if revealed
                if card["state"] == self.STATE_REVEALED:
                    self._draw_pattern(self.screen, card["pattern"], card_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(
            int(start_x + self.cursor_pos[0] * (card_width + card_gap) - 3),
            int(start_y + self.cursor_pos[1] * (card_height + card_gap) - 3),
            int(card_width + 6),
            int(card_height + 6)
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=7)

        # Draw mismatch/match flash
        if self.mismatch_timer > 0 and self.second_selection:
            self._draw_flash(self.first_selection, self.second_selection, self.COLOR_MISMATCH_FLASH, card_width, card_height, card_gap, start_x, start_y)
        

    def _draw_flash(self, pos1, pos2, color, card_w, card_h, gap, start_x, start_y):
        for pos in [pos1, pos2]:
            flash_rect = pygame.Rect(
                int(start_x + pos[0] * (card_w + gap)),
                int(start_y + pos[1] * (card_h + gap)),
                int(card_w),
                int(card_h)
            )
            flash_surface = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(flash_surface, color, flash_surface.get_rect(), border_radius=5)
            self.screen.blit(flash_surface, flash_rect.topleft)

    def _draw_pattern(self, surface, pattern_id, rect):
        shape_id = pattern_id % 8
        color_id = pattern_id // 8
        color = self.PATTERN_COLORS[color_id]
        
        center_x, center_y = rect.center
        size = min(rect.width, rect.height) * 0.35

        # Using gfxdraw for anti-aliasing
        if shape_id == 0:  # Circle
            pygame.gfxdraw.aacircle(surface, int(center_x), int(center_y), int(size), color)
            pygame.gfxdraw.filled_circle(surface, int(center_x), int(center_y), int(size), color)
        elif shape_id == 1:  # Square
            pts = [
                (center_x - size, center_y - size), (center_x + size, center_y - size),
                (center_x + size, center_y + size), (center_x - size, center_y + size)
            ]
            pygame.gfxdraw.aapolygon(surface, pts, color)
            pygame.gfxdraw.filled_polygon(surface, pts, color)
        elif shape_id == 2:  # Triangle
            pts = [
                (center_x, center_y - size),
                (center_x - size, center_y + size * 0.7),
                (center_x + size, center_y + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(surface, pts, color)
            pygame.gfxdraw.filled_polygon(surface, pts, color)
        elif shape_id == 3:  # X-Cross
            pygame.draw.line(surface, color, (center_x - size, center_y - size), (center_x + size, center_y + size), 4)
            pygame.draw.line(surface, color, (center_x - size, center_y + size), (center_x + size, center_y - size), 4)
        elif shape_id == 4:  # Diamond
            pts = [
                (center_x, center_y - size), (center_x + size, center_y),
                (center_x, center_y + size), (center_x - size, center_y)
            ]
            pygame.gfxdraw.aapolygon(surface, pts, color)
            pygame.gfxdraw.filled_polygon(surface, pts, color)
        elif shape_id == 5:  # Plus
            pygame.draw.line(surface, color, (center_x, center_y - size), (center_x, center_y + size), 4)
            pygame.draw.line(surface, color, (center_x - size, center_y), (center_x + size, center_y), 4)
        elif shape_id == 6:  # Hexagon
            pts = [(center_x + size * math.cos(math.pi / 3 * i), center_y + size * math.sin(math.pi / 3 * i)) for i in range(6)]
            pygame.gfxdraw.aapolygon(surface, pts, color)
            pygame.gfxdraw.filled_polygon(surface, pts, color)
        elif shape_id == 7:  # Star
            pts = []
            for i in range(10):
                r = size if i % 2 == 0 else size * 0.4
                angle = math.pi / 5 * i - math.pi / 2
                pts.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, pts, color)
            pygame.gfxdraw.filled_polygon(surface, pts, color)

    def _render_ui(self):
        ui_y = self.SCREEN_HEIGHT - 35
        
        # Matched Pairs Text
        total_pairs = (self.GRID_COLS * self.GRID_ROWS) // 2
        pairs_text = f"PAIRS: {self.matched_pairs_count} / {total_pairs}"
        text_surface = self.font_large.render(pairs_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (20, ui_y))
        
        # Steps Progress Bar
        bar_width = 250
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = ui_y + 2

        progress = self.steps / self.MAX_STEPS
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, bg_rect, border_radius=5)
        
        fill_width = max(0, min(bar_width, bar_width * progress))
        fill_rect = pygame.Rect(bar_x, bar_y, fill_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, fill_rect, border_radius=5)

        # Steps Text
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_small.render(steps_text, True, self.COLOR_UI_TEXT)
        steps_rect = steps_surf.get_rect(center=bg_rect.center)
        self.screen.blit(steps_surf, steps_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run pygame in headless mode
    
    env = GameEnv()
    env.validate_implementation()

    # To visualize the game, you would need a different setup
    # that creates a real pygame window and updates it in a loop.
    # The following is a basic example for a real window.
    
    # Comment out the os.environ line above to run this part
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Memory Match")
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
        
    #     if keys[pygame.K_SPACE]: action[1] = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         obs, info = env.reset()
        
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     window.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     env.clock.tick(30) # Limit frame rate for human play

    # env.close()