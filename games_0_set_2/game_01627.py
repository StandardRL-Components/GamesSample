
# Generated: 2025-08-28T02:11:37.411015
# Source Brief: brief_01627.md
# Brief Index: 1627

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to flip a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A memory puzzle game. Find all matching pairs of cards on the grid before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 4, 6
        self.NUM_PAIRS = 12
        self.TOTAL_CARDS = self.NUM_PAIRS * 2
        self.MAX_MOVES = 30
        self.MISMATCH_PAUSE_STEPS = 2
        self.MATCH_FLASH_STEPS = 3

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

        # Visuals
        self.COLOR_BG = (44, 62, 80)  # Dark blue-grey
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_CARD_BACK = (127, 140, 141)
        self.COLOR_CARD_FACE = (236, 240, 241)
        self.COLOR_CURSOR = (241, 196, 15) # Yellow
        self.COLOR_MATCH = (46, 204, 113) # Green
        self.COLOR_MISMATCH = (231, 76, 60) # Red
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_FADE = (0, 0, 0, 150)

        # Card colors for values
        self.CARD_VALUE_COLORS = [
            (26, 188, 156), (52, 152, 219), (155, 89, 182), (243, 156, 18),
            (211, 84, 0), (192, 57, 43), (41, 128, 185), (39, 174, 96),
            (230, 126, 34), (142, 68, 173), (22, 160, 133), (241, 196, 15)
        ]

        # Card sizing
        self.GRID_TOP_MARGIN = 80
        self.GRID_SIDE_MARGIN = 20
        grid_w = self.WIDTH - 2 * self.GRID_SIDE_MARGIN
        grid_h = self.HEIGHT - self.GRID_TOP_MARGIN - 20
        self.CARD_MARGIN = 8
        self.CARD_WIDTH = (grid_w - (self.GRID_COLS + 1) * self.CARD_MARGIN) / self.GRID_COLS
        self.CARD_HEIGHT = (grid_h - (self.GRID_ROWS + 1) * self.CARD_MARGIN) / self.GRID_ROWS

        # Fonts
        try:
            self.font_card = pygame.font.SysFont("Arial", int(self.CARD_HEIGHT * 0.6), bold=True)
            self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_end_game = pygame.font.SysFont("Arial", 60, bold=True)
        except pygame.error:
            self.font_card = pygame.font.Font(None, int(self.CARD_HEIGHT * 0.7))
            self.font_ui = pygame.font.Font(None, 32)
            self.font_end_game = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.cards = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = 0
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.second_selection = None
        self.mismatch_pause_counter = 0
        self.last_space_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [0, 0]  # [col, row]
        self.first_selection = None
        self.second_selection = None
        self.matched_pairs = 0
        self.mismatch_pause_counter = 0
        self.last_space_held = False
        
        self._setup_cards()
        
        return self._get_observation(), self._get_info()

    def _setup_cards(self):
        values = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(values)
        self.cards = []
        for i in range(self.TOTAL_CARDS):
            row = i // self.GRID_COLS
            col = i % self.GRID_COLS
            self.cards.append({
                "value": values[i], "id": i, "row": row, "col": col,
                "is_flipped": False, "is_matched": False,
                "just_flipped": False, "match_flash_counter": 0,
            })

    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        # If paused for mismatch, just decrement counter and return
        if self.mismatch_pause_counter > 0:
            self.mismatch_pause_counter -= 1
            if self.mismatch_pause_counter == 0:
                # Flip cards back and clear selections
                if self.first_selection: self.first_selection['is_flipped'] = False
                if self.second_selection: self.second_selection['is_flipped'] = False
                self.first_selection = None
                self.second_selection = None
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Process input only if not game over
        if not self.game_over:
            # Movement
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

            # Selection (on space press)
            if space_press and self.second_selection is None:
                card = self._get_card_at_cursor()
                if card and not card['is_flipped'] and not card['is_matched']:
                    card['is_flipped'] = True
                    card['just_flipped'] = True # For pop animation
                    
                    if self.first_selection is None:
                        self.first_selection = card
                    elif card['id'] != self.first_selection['id']:
                        self.second_selection = card
                        self.moves_left -= 1
                        
                        # Check for match
                        if self.first_selection['value'] == self.second_selection['value']:
                            # MATCH
                            reward += 10
                            self.score += 10
                            self.first_selection['is_matched'] = True
                            self.second_selection['is_matched'] = True
                            self.first_selection['match_flash_counter'] = self.MATCH_FLASH_STEPS
                            self.second_selection['match_flash_counter'] = self.MATCH_FLASH_STEPS
                            self.matched_pairs += 1
                            self.first_selection = None
                            self.second_selection = None
                            # Sound: Correct match
                        else:
                            # MISMATCH
                            reward -= 1
                            self.score -= 1
                            self.mismatch_pause_counter = self.MISMATCH_PAUSE_STEPS
                            # Sound: Incorrect match
        
        self.steps += 1
        
        # Check for termination
        if not self.game_over:
            if self.matched_pairs == self.NUM_PAIRS:
                self.game_over = True
                self.win_state = True
                terminated = True
                reward += 100
                self.score += 100
                # Sound: Victory
            elif self.moves_left <= 0:
                self.game_over = True
                self.win_state = False
                terminated = True
                reward -= 50
                self.score -= 50
                # Sound: Failure

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_card_at_cursor(self):
        for card in self.cards:
            if card['col'] == self.cursor_pos[0] and card['row'] == self.cursor_pos[1]:
                return card
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Reset one-frame animation flags
        for card in self.cards:
            card['just_flipped'] = False

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for card in self.cards:
            card_x = self.GRID_SIDE_MARGIN + self.CARD_MARGIN + card['col'] * (self.CARD_WIDTH + self.CARD_MARGIN)
            card_y = self.GRID_TOP_MARGIN + self.CARD_MARGIN + card['row'] * (self.CARD_HEIGHT + self.CARD_MARGIN)
            
            pop_offset = 0
            if card['just_flipped']:
                pop_offset = 5 # "Pop" effect
            
            card_rect = pygame.Rect(
                card_x - pop_offset, card_y - pop_offset,
                self.CARD_WIDTH + 2 * pop_offset, self.CARD_HEIGHT + 2 * pop_offset
            )

            # Draw card
            if card['is_flipped'] or card['is_matched']:
                pygame.draw.rect(self.screen, self.COLOR_CARD_FACE, card_rect, border_radius=5)
                
                value_color = self.CARD_VALUE_COLORS[card['value'] % len(self.CARD_VALUE_COLORS)]
                text_surf = self.font_card.render(str(card['value'] + 1), True, value_color)
                text_rect = text_surf.get_rect(center=card_rect.center)
                self.screen.blit(text_surf, text_rect)
            else:
                pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, card_rect, border_radius=5)
                # Simple back design
                pygame.draw.rect(self.screen, self.COLOR_GRID, card_rect.inflate(-10, -10), 2, border_radius=3)
            
            # Matched card visual feedback
            if card['is_matched']:
                fade_surf = pygame.Surface(card_rect.size, pygame.SRCALPHA)
                alpha = 200 - card['match_flash_counter'] * (200 / self.MATCH_FLASH_STEPS)
                fade_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], alpha))
                self.screen.blit(fade_surf, card_rect.topleft)

            # Mismatch/Match flash
            if self.mismatch_pause_counter > 0 and card['id'] in [c['id'] for c in [self.first_selection, self.second_selection] if c]:
                pygame.gfxdraw.box(self.screen, card_rect, (*self.COLOR_MISMATCH, 100))
            if card['match_flash_counter'] > 0:
                card['match_flash_counter'] -= 1
                pygame.gfxdraw.box(self.screen, card_rect, (*self.COLOR_MATCH, 100))

        # Draw cursor
        cursor_x = self.GRID_SIDE_MARGIN + self.CARD_MARGIN + self.cursor_pos[0] * (self.CARD_WIDTH + self.CARD_MARGIN)
        cursor_y = self.GRID_TOP_MARGIN + self.CARD_MARGIN + self.cursor_pos[1] * (self.CARD_HEIGHT + self.CARD_MARGIN)
        cursor_rect = pygame.Rect(cursor_x - 4, cursor_y - 4, self.CARD_WIDTH + 8, self.CARD_HEIGHT + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)

    def _render_ui(self):
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (20, 20))

        # Pairs Found
        pairs_text = f"Pairs: {self.matched_pairs} / {self.NUM_PAIRS}"
        pairs_surf = self.font_ui.render(pairs_text, True, self.COLOR_TEXT)
        pairs_rect = pairs_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(pairs_surf, pairs_rect)
        
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(midtop=(self.WIDTH // 2, 20))
        self.screen.blit(score_surf, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_FADE)
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win_state else "GAME OVER"
            end_color = self.COLOR_MATCH if self.win_state else self.COLOR_MISMATCH
            end_surf = self.font_end_game.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "matched_pairs": self.matched_pairs,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Example ---
    # This part requires a display. Set render_mode='human' if you adapt the class for it.
    # For now, we'll simulate it and print info.
    
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    # You would need a pygame window and event loop to play this manually.
    # The following is a demonstration of the API.
    print("Running a short random-action demonstration...")
    for i in range(100):
        if terminated:
            print(f"Episode finished after {i+1} steps.")
            break
            
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # print(f"Step {i}: Action: {action}, Reward: {reward}, Info: {info}")

    print(f"Demonstration finished. Total reward: {total_reward}")
    env.close()