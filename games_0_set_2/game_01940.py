
# Generated: 2025-08-27T18:44:54.597499
# Source Brief: brief_01940.md
# Brief Index: 1940

        
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


class Card:
    """Helper class to store card state."""
    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.state = "hidden"  # "hidden", "revealed", "matched"
        self.match_animation_timer = 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selection. Space to reveal a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Test your memory by revealing matching number pairs on a grid. You lose after 3 incorrect guesses."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_large = pygame.font.Font(None, 80)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID_BG = (40, 50, 60)
        self.COLOR_CARD_HIDDEN = (70, 85, 100)
        self.COLOR_CARD_REVEALED = (210, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_MATCH = (60, 220, 120)
        self.COLOR_MISMATCH = (255, 80, 80)
        self.COLOR_HEART = (220, 40, 40)
        
        # Game Constants
        self.GRID_ROWS = 4
        self.GRID_COLS = 4
        self.MAX_INCORRECT_GUESSES = 3
        self.MAX_STEPS = 1000
        self.MISMATCH_DELAY_STEPS = 3 # Number of steps to show mismatched cards
        self.MATCH_ANIMATION_STEPS = 5

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selection = []
        self.incorrect_guesses = 0
        self.matches_found = 0
        self.mismatch_timer = 0
        self.mismatched_pair = []
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.incorrect_guesses = 0
        self.matches_found = 0
        
        self.cursor_pos = [0, 0]
        self.selection = []
        self.mismatch_timer = 0
        self.mismatched_pair = []
        self.prev_space_held = False

        # Create and shuffle cards
        numbers = list(range(1, (self.GRID_ROWS * self.GRID_COLS) // 2 + 1)) * 2
        self.np_random.shuffle(numbers)
        
        self.grid = []
        for r in range(self.GRID_ROWS):
            row_list = []
            for c in range(self.GRID_COLS):
                card_value = numbers.pop()
                row_list.append(Card(card_value, r, c))
            self.grid.append(row_list)
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # First, handle any automatic state transitions (like flipping cards back)
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                for card_pos in self.mismatched_pair:
                    self.grid[card_pos[0]][card_pos[1]].state = "hidden"
                self.mismatched_pair = []
        else:
            # Unpack and process player action
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Movement ---
            if movement == 1: # Up
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: # Down
                self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
            elif movement == 3: # Left
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: # Right
                self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

            # --- Handle Selection ---
            card_selected = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
            is_new_press = space_held and not self.prev_space_held

            if is_new_press and card_selected.state == "hidden" and len(self.selection) < 2:
                # SFX: card_flip.wav
                card_selected.state = "revealed"
                self.selection.append(card_selected)

                if len(self.selection) == 2:
                    card1, card2 = self.selection[0], self.selection[1]
                    
                    # Check for match
                    if card1.value == card2.value:
                        # SFX: match_success.wav
                        reward = 10
                        self.score += reward
                        self.matches_found += 1
                        card1.state = "matched"
                        card2.state = "matched"
                        card1.match_animation_timer = self.MATCH_ANIMATION_STEPS
                        card2.match_animation_timer = self.MATCH_ANIMATION_STEPS
                        self.selection = []
                    else:
                        # SFX: match_fail.wav
                        reward = -1
                        self.score += reward
                        self.incorrect_guesses += 1
                        self.mismatch_timer = self.MISMATCH_DELAY_STEPS
                        self.mismatched_pair = [(card1.row, card1.col), (card2.row, card2.col)]
                        self.selection = []

            self.prev_space_held = space_held

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _check_termination(self):
        if self.game_over: # Already terminated
            return True
        
        if self.matches_found == (self.GRID_ROWS * self.GRID_COLS) / 2:
            self.game_over = True
        elif self.incorrect_guesses >= self.MAX_INCORRECT_GUESSES:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Define grid layout
        grid_width = 440
        grid_height = 320
        start_x = (640 - grid_width) / 2
        start_y = (400 - grid_height) / 2 + 20
        card_size = 80
        padding = 20

        # Draw grid background
        grid_rect = pygame.Rect(start_x - padding, start_y - padding, grid_width + 2*padding, grid_height + 2*padding)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=15)
        
        # Draw cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                card_x = start_x + c * (card_size + padding)
                card_y = start_y + r * (card_size + padding)
                card_rect = pygame.Rect(card_x, card_y, card_size, card_size)
                
                color = self.COLOR_CARD_HIDDEN
                if card.state == "revealed":
                    color = self.COLOR_CARD_REVEALED
                elif card.state == "matched":
                    color = self.COLOR_MATCH

                # Highlight mismatched cards
                if self.mismatch_timer > 0 and (r, c) in self.mismatched_pair:
                    color = self.COLOR_MISMATCH

                pygame.draw.rect(self.screen, color, card_rect, border_radius=8)

                if card.state != "hidden":
                    num_text = self.font_medium.render(str(card.value), True, self.COLOR_BG)
                    text_rect = num_text.get_rect(center=card_rect.center)
                    self.screen.blit(num_text, text_rect)
                
                # Draw match animation
                if card.match_animation_timer > 0:
                    card.match_animation_timer -= 1
                    s = self.MATCH_ANIMATION_STEPS
                    t = card.match_animation_timer
                    alpha = int(255 * (t / s))
                    radius = int(card_size * 0.75 * (1 - (t / s)))
                    pygame.gfxdraw.aacircle(self.screen, int(card_rect.centerx), int(card_rect.centery), radius, (*self.COLOR_MATCH, alpha))
                    pygame.gfxdraw.aacircle(self.screen, int(card_rect.centerx), int(card_rect.centery), radius-1, (*self.COLOR_MATCH, alpha))

        # Draw cursor
        cursor_x = start_x + self.cursor_pos[1] * (card_size + padding)
        cursor_y = start_y + self.cursor_pos[0] * (card_size + padding)
        cursor_rect = pygame.Rect(cursor_x - 5, cursor_y - 5, card_size + 10, card_size + 10)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=12)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Matches
        matches_text = self.font_small.render(f"Matches: {self.matches_found} / {self.GRID_ROWS * self.GRID_COLS // 2}", True, self.COLOR_TEXT)
        matches_rect = matches_text.get_rect(centerx=640/2)
        matches_rect.top = 15
        self.screen.blit(matches_text, matches_rect)

        # Lives (Incorrect Guesses)
        lives_text = self.font_small.render("Lives:", True, self.COLOR_TEXT)
        lives_rect = lives_text.get_rect(right=620, top=15)
        self.screen.blit(lives_text, (lives_rect.left - 120, 15))
        for i in range(self.MAX_INCORRECT_GUESSES):
            heart_pos = (lives_rect.left - 80 + i * 35, 29)
            if i < self.MAX_INCORRECT_GUESSES - self.incorrect_guesses:
                self._draw_heart(self.screen, heart_pos)
            else:
                self._draw_heart(self.screen, heart_pos, color=(80, 80, 80))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.matches_found == (self.GRID_ROWS * self.GRID_COLS) / 2:
                msg = "YOU WIN!"
                color = self.COLOR_MATCH
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISMATCH
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(640/2, 400/2))
            self.screen.blit(end_text, end_rect)

    def _draw_heart(self, surface, pos, color=None):
        if color is None:
            color = self.COLOR_HEART
        x, y = pos
        # A simple heart shape using polygons
        points = [
            (x, y-5), (x+5, y-10), (x+10, y-5), (x, y+5), 
            (x-10, y-5), (x-5, y-10)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.gfxdraw.aapolygon(surface, points, color)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "matches_found": self.matches_found,
            "incorrect_guesses": self.incorrect_guesses,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Number Memory")
    
    running = True
    total_reward = 0
    
    # Game loop for human play
    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(100)

    env.close()