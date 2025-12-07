
# Generated: 2025-08-28T04:34:57.836345
# Source Brief: brief_02369.md
# Brief Index: 2369

        
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
    A vibrant, grid-based memory matching game.

    The player controls a cursor on an 8x4 grid of face-down cards. The goal is to
    select two cards that match. A correct match leaves the cards face-up and
    awards points. An incorrect match flips the cards back down after a short
    delay and counts as a mistake. The game ends when all 16 pairs are found (win)
    or when the player makes 3 incorrect matches (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to flip a card."
    )

    # Short, user-facing description of the game
    game_description = (
        "A vibrant memory game. Find all the matching pairs of colored cards before you run out of lives."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """Initializes the memory game environment."""
        super().__init__()
        
        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_COLS, self.GRID_ROWS = 8, 4
        self.NUM_PAIRS = (self.GRID_COLS * self.GRID_ROWS) // 2
        self.MAX_INCORRECT = 3
        self.MAX_STEPS = 1000
        self.MISMATCH_COOLDOWN_FRAMES = 25

        # Visuals
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_CARD_BACK = (70, 80, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.CARD_COLORS = self._generate_card_colors()
        self.font_main = pygame.font.Font(None, 36)
        self.card_w = 0
        self.card_h = 0

        # Game state variables (to be initialized in reset)
        self.grid = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.steps = 0
        self.incorrect_matches = 0
        self.matched_pairs = 0
        self.game_over = False
        self.first_selection = None
        self.second_selection = None
        self.mismatch_cooldown = 0
        self.particles = []
        self.last_space_held = False

        # Initialize state for the first time
        self.reset()

        # Run validation check
        self.validate_implementation()

    def _generate_card_colors(self):
        """Generates a list of visually distinct, vibrant colors for the cards."""
        colors = []
        for i in range(self.NUM_PAIRS):
            hue = int((i / self.NUM_PAIRS) * 360)
            color = pygame.Color(0)
            color.hsla = (hue, 100, 60, 100)
            colors.append(tuple(color)[:3])
        return colors

    def reset(self, seed=None, options=None):
        """Resets the game to its initial state."""
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.incorrect_matches = 0
        self.matched_pairs = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.second_selection = None
        self.mismatch_cooldown = 0
        self.particles = []
        self.last_space_held = False

        # Create and shuffle cards
        card_ids = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(card_ids)

        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                card_id = card_ids.pop(0)
                card = {
                    "id": card_id,
                    "state": "hidden",  # hidden, revealed, matched
                    "pos": (c, r),
                    "anim_progress": 0.0,
                    "anim_dir": 0,  # 0: none, 1: revealing, -1: hiding
                }
                row.append(card)
            self.grid.append(row)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """Processes an action and updates the game state for one step."""
        reward = 0.0
        self.game_over = False

        self._update_animations()

        # Handle mismatch cooldown period
        if self.mismatch_cooldown > 0:
            self.mismatch_cooldown -= 1
            if self.mismatch_cooldown == 0:
                # Flip mismatched cards back
                self.first_selection['anim_dir'] = -1
                self.second_selection['anim_dir'] = -1
                self.first_selection = None
                self.second_selection = None
        
        # Process actions only if not in a forced wait (mismatch cooldown)
        if self.mismatch_cooldown == 0:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Movement ---
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            if movement == 2: self.cursor_pos[1] += 1  # Down
            if movement == 3: self.cursor_pos[0] -= 1  # Left
            if movement == 4: self.cursor_pos[0] += 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # --- Handle Selection (on space key press) ---
            space_pressed = space_held and not self.last_space_held
            if space_pressed:
                card = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
                
                # Can only select a hidden card
                if card['state'] == 'hidden':
                    card['state'] = 'revealed'
                    card['anim_dir'] = 1  # Start revealing animation
                    reward += 0.1  # Reward for revealing a new card
                    # sfx: card_flip.wav

                    if self.first_selection is None:
                        self.first_selection = card
                    elif self.second_selection is None:
                        self.second_selection = card
                        
                        # Check for a match
                        if self.first_selection['id'] == self.second_selection['id']:
                            # MATCH
                            self.first_selection['state'] = 'matched'
                            self.second_selection['state'] = 'matched'
                            self.score += 10
                            reward += 10
                            self.matched_pairs += 1
                            self._create_particles(self.first_selection['pos'])
                            self._create_particles(self.second_selection['pos'])
                            self.first_selection = None
                            self.second_selection = None
                            # sfx: match_success.wav
                        else:
                            # MISMATCH
                            self.score -= 5
                            reward -= 5
                            self.incorrect_matches += 1
                            self.mismatch_cooldown = self.MISMATCH_COOLDOWN_FRAMES
                            # sfx: match_fail.wav

        self.last_space_held = action[1] == 1
        
        self._update_particles()

        # Check termination conditions
        if self.matched_pairs == self.NUM_PAIRS:
            self.game_over = True
            reward += 100
            self.score += 100
            # sfx: game_win.wav
        elif self.incorrect_matches >= self.MAX_INCORRECT:
            self.game_over = True
            reward -= 100
            self.score -= 100
            # sfx: game_over.wav
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated is always False
            self._get_info()
        )

    def _update_animations(self):
        """Updates the progress of all card flip animations."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                if card['anim_dir'] != 0:
                    card['anim_progress'] += card['anim_dir'] * 0.1  # Animation speed
                    if card['anim_progress'] >= 1.0:
                        card['anim_progress'] = 1.0
                        card['anim_dir'] = 0
                    elif card['anim_progress'] <= 0.0:
                        card['anim_progress'] = 0.0
                        card['anim_dir'] = 0
                        if card['state'] == 'revealed':
                           card['state'] = 'hidden'

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the game board, cards, cursor, and particles."""
        margin = 20
        grid_w = self.screen.get_width() - 2 * margin
        grid_h = self.screen.get_height() - 80  # Space for UI
        self.card_w = (grid_w - (self.GRID_COLS - 1) * 5) / self.GRID_COLS
        self.card_h = (grid_h - (self.GRID_ROWS - 1) * 5) / self.GRID_ROWS
        
        # Render cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                card_x = margin + c * (self.card_w + 5)
                card_y = 60 + r * (self.card_h + 5)
                
                # Smooth flip animation using cosine interpolation
                scale_factor = (1 - math.cos(card['anim_progress'] * math.pi)) / 2
                current_w = self.card_w * abs(1 - 2 * scale_factor)
                is_flipping_to_front = scale_factor > 0.5

                card_rect = pygame.Rect(
                    card_x + (self.card_w - current_w) / 2,
                    card_y,
                    max(0, current_w),
                    max(0, self.card_h)
                )

                if card['state'] == 'hidden' and not is_flipping_to_front:
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, card_rect, border_radius=5)
                else:  # revealed, matched, or flipping to front
                    color = self.CARD_COLORS[card['id']]
                    if card['state'] == 'matched':
                        h, s, l, a = pygame.Color(*color).hsla
                        desaturated_color = pygame.Color(0)
                        desaturated_color.hsla = (h, s * 0.3, l * 0.8, a)
                        pygame.draw.rect(self.screen, desaturated_color, card_rect, border_radius=5)
                        cx, cy = card_rect.center
                        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), 10, (255, 255, 255, 100))
                    else:  # revealed
                        pygame.draw.rect(self.screen, color, card_rect, border_radius=5)

        # Render cursor with a pulsating glow effect
        cursor_x = margin + self.cursor_pos[0] * (self.card_w + 5)
        cursor_y = 60 + self.cursor_pos[1] * (self.card_h + 5)
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 3
        cursor_rect = pygame.Rect(cursor_x - 4, cursor_y - 4, self.card_w + 8, self.card_h + 8)
        glow_rect = pygame.Rect(cursor_x - 4 - pulse, cursor_y - 4 - pulse, self.card_w + 8 + 2*pulse, self.card_h + 8 + 2*pulse)
        
        pygame.draw.rect(self.screen, (255, 255, 0, 50), glow_rect, 2, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], max(0, alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), color_with_alpha)

    def _render_ui(self):
        """Renders the score and remaining lives."""
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen.get_width() - score_text.get_width() - 15, 15))

        lives_text = self.font_main.render("Lives:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (15, 15))
        bar_w, bar_h = 40, 20
        for i in range(self.MAX_INCORRECT):
            bar_x = 100 + i * (bar_w + 5)
            bar_y = 17
            color = (50, 200, 50) if i < self.MAX_INCORRECT - self.incorrect_matches else (200, 50, 50)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=3)

    def _create_particles(self, grid_pos):
        """Creates a burst of particles at a card's location."""
        margin = 20
        c, r = grid_pos
        center_x = margin + c * (self.card_w + 5) + self.card_w / 2
        center_y = 60 + r * (self.card_h + 5) + self.card_h / 2
        card_id = self.grid[r][c]['id']
        color = self.CARD_COLORS[card_id]
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.np_random.uniform(3, 7),
                'life': 30, 'max_life': 30, 'color': color
            })

    def _update_particles(self):
        """Updates the position and lifetime of all active particles."""
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] *= 0.98
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        """Returns a dictionary with the current score and step count."""
        return {"score": self.score, "steps": self.steps}
    
    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")