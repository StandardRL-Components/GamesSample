
# Generated: 2025-08-28T00:50:02.293331
# Source Brief: brief_03916.md
# Brief Index: 3916

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a fraction, then a pie chart to match. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game where you match numerical fractions to their visual pie-chart representations. Win by making 5 correct matches, but be careful - 3 wrong matches and you lose!"
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Visuals & Layout
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 22, bold=True)
        
        self.colors = {
            "bg": (25, 28, 36),
            "grid": (45, 52, 64),
            "cell": (35, 40, 50),
            "text": (220, 220, 230),
            "pie_fill": (76, 201, 240),
            "pie_empty": (50, 58, 71),
            "cursor": (255, 198, 0),
            "selected": (0, 150, 255),
            "correct": (46, 204, 113),
            "incorrect": (231, 76, 60),
            "matched_overlay": (0, 0, 0, 128),
        }

        self.grid_cols, self.grid_rows = 5, 2
        self.grid_margin_x = 20
        self.grid_margin_y = 80
        self.cell_gap = 8
        self.cell_width = (self.screen_width - 2 * self.grid_margin_x - (self.grid_cols - 1) * self.cell_gap) / self.grid_cols
        self.cell_height = (self.screen_height - self.grid_margin_y - self.grid_margin_x) / self.grid_rows

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.grid_items = []
        self.selected_item_idx = -1
        self.correct_matches = 0
        self.incorrect_matches = 0
        self.feedback_animations = []
        self.win_state = ""

        # Possible fractions to generate
        self.possible_fractions = [
            (1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5),
            (1, 6), (5, 6), (1, 8), (3, 8), (5, 8), (7, 8)
        ]

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.selected_item_idx = -1
        self.correct_matches = 0
        self.incorrect_matches = 0
        self.feedback_animations = []
        self.win_state = ""
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.grid_items = []
        num_pairs = 5
        
        # Choose 5 unique fractions
        fraction_indices = self.np_random.choice(len(self.possible_fractions), num_pairs, replace=False)
        chosen_fractions = [self.possible_fractions[i] for i in fraction_indices]
        
        # Create item pairs
        for i, frac in enumerate(chosen_fractions):
            self.grid_items.append({'id': i, 'type': 'text', 'value': frac, 'pos': (0,0), 'matched': False})
            self.grid_items.append({'id': i, 'type': 'pie', 'value': frac, 'pos': (0,0), 'matched': False})
            
        # Shuffle positions
        positions = [(c, r) for c in range(self.grid_cols) for r in range(self.grid_rows)]
        self.np_random.shuffle(positions)
        
        for i, item in enumerate(self.grid_items):
            item['pos'] = positions[i]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Update cursor position
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1)) # Up
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.grid_rows - 1, self.cursor_pos[1] + 1)) # Down
        elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1]) # Left
        elif movement == 4: self.cursor_pos = (min(self.grid_cols - 1, self.cursor_pos[0] + 1), self.cursor_pos[1]) # Right

        # 2. Handle actions
        if shift_pressed and self.selected_item_idx != -1:
            self.selected_item_idx = -1
            # sfx: deselect sound
        
        if space_pressed:
            reward += self._handle_selection()
            # sfx: selection sound or match sound

        # 3. Update game logic
        self.steps += 1
        self._update_animations()
        
        # 4. Check for termination
        terminated = False
        if self.correct_matches >= 5:
            terminated = True
            self.win_state = "YOU WIN!"
            reward += 100
            # sfx: win fanfare
        elif self.incorrect_matches >= 3:
            terminated = True
            self.win_state = "GAME OVER"
            reward -= 100
            # sfx: lose sound
        elif self.steps >= 1000:
            terminated = True
            self.win_state = "TIME UP"
            reward -= 50

        if terminated:
            self.game_over = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_selection(self):
        cursor_item_idx, cursor_item = self._get_item_at_cursor()
        if cursor_item is None or cursor_item['matched']:
            return 0

        # State 1: Nothing is selected
        if self.selected_item_idx == -1:
            if cursor_item['type'] == 'text':
                self.selected_item_idx = cursor_item_idx
                return 0.1 # Small reward for valid selection
            else:
                return -0.1 # Penalty for trying to select a pie first
        
        # State 2: A text item is selected
        else:
            selected_item = self.grid_items[self.selected_item_idx]
            
            # Trying to select another text item -> switch selection
            if cursor_item['type'] == 'text':
                if self.selected_item_idx != cursor_item_idx:
                    self.selected_item_idx = cursor_item_idx
                return 0
            
            # Trying to select a pie item -> check for match
            elif cursor_item['type'] == 'pie':
                # Correct Match
                if selected_item['id'] == cursor_item['id']:
                    self.correct_matches += 1
                    selected_item['matched'] = True
                    cursor_item['matched'] = True
                    self._add_feedback_animation(cursor_item['pos'], self.colors['correct'])
                    self._add_feedback_animation(selected_item['pos'], self.colors['correct'])
                    self.selected_item_idx = -1
                    return 10
                # Incorrect Match
                else:
                    self.incorrect_matches += 1
                    self._add_feedback_animation(cursor_item['pos'], self.colors['incorrect'])
                    self._add_feedback_animation(selected_item['pos'], self.colors['incorrect'])
                    self.selected_item_idx = -1
                    return -10
        return 0

    def _get_item_at_cursor(self):
        for i, item in enumerate(self.grid_items):
            if item['pos'] == self.cursor_pos:
                return i, item
        return -1, None

    def _add_feedback_animation(self, pos, color):
        x = self.grid_margin_x + pos[0] * (self.cell_width + self.cell_gap) + self.cell_width / 2
        y = self.grid_margin_y + pos[1] * (self.cell_height + self.cell_gap) + self.cell_height / 2
        self.feedback_animations.append({'pos': (x, y), 'color': color, 'radius': 0, 'alpha': 255})

    def _update_animations(self):
        for anim in self.feedback_animations:
            anim['radius'] += 6
            anim['alpha'] -= 15
        self.feedback_animations = [a for a in self.feedback_animations if a['alpha'] > 0]

    def _get_observation(self):
        self.screen.fill(self.colors['bg'])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render grid items
        for i, item in enumerate(self.grid_items):
            col, row = item['pos']
            cell_rect = pygame.Rect(
                self.grid_margin_x + col * (self.cell_width + self.cell_gap),
                self.grid_margin_y + row * (self.cell_height + self.cell_gap),
                self.cell_width, self.cell_height
            )
            pygame.draw.rect(self.screen, self.colors['cell'], cell_rect, border_radius=8)

            center = cell_rect.center
            if item['type'] == 'text':
                num, den = item['value']
                text_str = f"{num} / {den}"
                text_surf = self.font_medium.render(text_str, True, self.colors['text'])
                self.screen.blit(text_surf, text_surf.get_rect(center=center))
            elif item['type'] == 'pie':
                self._draw_pie(self.screen, center, min(self.cell_width, self.cell_height) * 0.35, item['value'])

            if item['matched']:
                s = pygame.Surface((self.cell_width, self.cell_height), pygame.SRCALPHA)
                s.fill(self.colors['matched_overlay'])
                self.screen.blit(s, cell_rect.topleft)
        
        # Render highlights
        self._render_highlights()
        
        # Render animations
        self._render_animations()

    def _render_highlights(self):
        # Render cursor
        cursor_col, cursor_row = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_margin_x + cursor_col * (self.cell_width + self.cell_gap) - 4,
            self.grid_margin_y + cursor_row * (self.cell_height + self.cell_gap) - 4,
            self.cell_width + 8, self.cell_height + 8
        )
        pygame.draw.rect(self.screen, self.colors['cursor'], cursor_rect, 4, border_radius=12)

        # Render selected item
        if self.selected_item_idx != -1:
            item = self.grid_items[self.selected_item_idx]
            sel_col, sel_row = item['pos']
            selected_rect = pygame.Rect(
                self.grid_margin_x + sel_col * (self.cell_width + self.cell_gap) - 4,
                self.grid_margin_y + sel_row * (self.cell_height + self.cell_gap) - 4,
                self.cell_width + 8, self.cell_height + 8
            )
            pygame.draw.rect(self.screen, self.colors['selected'], selected_rect, 4, border_radius=12)

    def _render_animations(self):
        for anim in self.feedback_animations:
            if anim['alpha'] > 0:
                color = (*anim['color'], anim['alpha'])
                s = pygame.Surface((anim['radius']*2, anim['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (anim['radius'], anim['radius']), anim['radius'])
                self.screen.blit(s, s.get_rect(center=anim['pos']))

    def _draw_pie(self, surface, center, radius, fraction):
        num, den = fraction
        
        # Draw empty part
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), self.colors['pie_empty'])
        
        # Draw filled part
        angle_per_slice = 360 / den
        start_angle_rad = math.radians(90)
        
        points = [center]
        for i in range(num + 1):
            angle = start_angle_rad - math.radians(i * angle_per_slice)
            x = center[0] + radius * math.cos(angle)
            y = center[1] - radius * math.sin(angle)
            points.append((x, y))
        
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(surface, points, self.colors['pie_fill'])
            pygame.gfxdraw.aapolygon(surface, points, self.colors['pie_fill'])

        # Draw outline
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), self.colors['text'])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.colors['text'])
        self.screen.blit(score_text, (15, 10))

        # Progress indicators
        indicator_y = 45
        indicator_radius = 8
        
        # Correct matches
        for i in range(5):
            color = self.colors['correct'] if i < self.correct_matches else self.colors['grid']
            pygame.draw.circle(self.screen, color, (25 + i * 25, indicator_y), indicator_radius)
            pygame.draw.circle(self.screen, self.colors['text'], (25 + i * 25, indicator_y), indicator_radius, 1)

        # Incorrect matches
        for i in range(3):
            color = self.colors['incorrect'] if i < self.incorrect_matches else self.colors['grid']
            pygame.draw.circle(self.screen, color, (self.screen_width - 25 - i * 25, indicator_y), indicator_radius)
            pygame.draw.circle(self.screen, self.colors['text'], (self.screen_width - 25 - i * 25, indicator_y), indicator_radius, 1)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = self.font_large.render(self.win_state, True, self.colors['text'])
            self.screen.blit(win_text, win_text.get_rect(center=(self.screen_width/2, self.screen_height/2)))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "correct_matches": self.correct_matches,
            "incorrect_matches": self.incorrect_matches,
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    while not terminated:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # In human-play mode, we need a display
        if env.metadata["render_modes"]:
            # Create a window if it doesn't exist
            if not hasattr(env, 'window'):
                env.window = pygame.display.set_mode((env.screen_width, env.screen_height))
                pygame.display.set_caption("Fraction Matcher")
            
            # Since auto_advance is False, we only step when an action occurs
            # For a better human experience, we'll step even on no-op if a key is pressed.
            # A simple way is to just step on every frame for human play.
            obs, reward, terminated, truncated, info = env.step(action)

            # Blit the observation to the window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            env.window.blit(surf, (0, 0))
            pygame.display.flip()

            env.clock.tick(15) # Limit FPS for human play
            
            if terminated:
                pygame.time.wait(2000) # Pause on game over screen

    env.close()