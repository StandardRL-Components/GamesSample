
# Generated: 2025-08-28T06:56:27.104743
# Source Brief: brief_03089.md
# Brief Index: 3089

        
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
    A fast-paced, procedurally generated grid-based memory matching game where
    players race against time to reveal and match pairs of visually stunning patterns.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to flip a card."
    )

    # Short, user-facing description of the game
    game_description = (
        "A memory matching game against the clock. Find all the pairs of "
        "colorful patterns before time runs out!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GRID_ROWS = 4
    GRID_COLS = 4
    CARD_COUNT = GRID_ROWS * GRID_COLS
    PAIR_COUNT = CARD_COUNT // 2

    # Colors
    COLOR_BG = (26, 26, 46) # Dark blue
    COLOR_GRID = (15, 33, 62)
    COLOR_CARD_DOWN = (15, 52, 96)
    COLOR_CARD_BORDER = (240, 240, 240)
    COLOR_CURSOR = (255, 215, 0) # Gold
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIME_WARN = (255, 80, 80)
    
    PATTERN_COLORS = [
        (255, 89, 94), (255, 202, 58), (138, 201, 38), (25, 130, 196),
        (106, 76, 147), (255, 127, 80), (30, 144, 255), (221, 160, 221)
    ]

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
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.card_patterns = []
        self.cards = []
        self.cursor_pos = [0, 0]
        self.revealed_cards = []
        self.mismatch_timer = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.last_space_state = 0
        self.move_cooldown = 0

        self.reset()
        # self.validate_implementation() # Optional: for debugging during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 60 * self.FPS # 60 seconds
        self.cursor_pos = [0, 0]
        self.revealed_cards = []
        self.mismatch_timer = 0
        self.particles = []
        self.last_space_state = 0
        self.move_cooldown = 0
        
        self._setup_cards()
        self._pre_render_patterns()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small time penalty to encourage speed

        # --- Handle Input ---
        reward += self._handle_input(action)
        
        # --- Update Game State ---
        self._update_game_state()
        self._update_animations()
        
        # --- Termination and Final Rewards ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.time_remaining <= 0:
                reward -= 50 # Time out penalty
            elif self._all_cards_matched():
                reward += 50 # Win bonus
        
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if self.move_cooldown == 0:
            moved = False
            if movement == 1: # Up
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                moved = True
            elif movement == 2: # Down
                self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
                moved = True
            elif movement == 3: # Left
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                moved = True
            elif movement == 4: # Right
                self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
                moved = True
            if moved:
                self.move_cooldown = 5 # 5-frame cooldown for movement

        # Card selection (Space press)
        space_press = space_held and not self.last_space_state
        if space_press:
            card_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
            card = self.cards[card_idx]
            
            if card['state'] == 'down' and len(self.revealed_cards) < 2 and self.mismatch_timer == 0:
                card['state'] = 'flipping_up'
                card['flip_progress'] = 0
                self.revealed_cards.append(card)
                # Sound: card_flip.wav

                if len(self.revealed_cards) == 1:
                    reward += 0.1 # Reward for revealing a new card
                
                if len(self.revealed_cards) == 2:
                    card1, card2 = self.revealed_cards
                    if card1['pattern_id'] == card2['pattern_id']:
                        # Match
                        card1['state'] = 'matched'
                        card2['state'] = 'matched'
                        self.revealed_cards = []
                        reward += 10 # Match reward
                        self._create_particles(card1['rect'].center, card1['pattern_id'])
                        self._create_particles(card2['rect'].center, card2['pattern_id'])
                        # Sound: match_success.wav
                    else:
                        # Mismatch
                        self.mismatch_timer = self.FPS // 2 # 0.5s delay
                        reward += 0.1 # Reward for revealing a non-matching card
                        # Sound: match_fail.wav
            elif card['state'] == 'up':
                reward -= 0.1 # Penalty for re-selecting an already revealed card

        self.last_space_state = space_held
        return reward

    def _update_game_state(self):
        self.time_remaining = max(0, self.time_remaining - 1)
        self.move_cooldown = max(0, self.move_cooldown - 1)

        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0 and len(self.revealed_cards) == 2:
                for card in self.revealed_cards:
                    card['state'] = 'flipping_down'
                    card['flip_progress'] = 0
                self.revealed_cards = []
    
    def _update_animations(self):
        # Card flips
        for card in self.cards:
            if card['state'] == 'flipping_up':
                card['flip_progress'] += 0.2
                if card['flip_progress'] >= 1:
                    card['flip_progress'] = 1
                    card['state'] = 'up'
            elif card['state'] == 'flipping_down':
                card['flip_progress'] += 0.2
                if card['flip_progress'] >= 1:
                    card['flip_progress'] = 1
                    card['state'] = 'down'

        # Particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_cards()
        self._draw_cursor()
        self._draw_particles()

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Time
        time_sec = self.time_remaining / self.FPS
        time_color = self.COLOR_TIME_WARN if time_sec < 10 else self.COLOR_TEXT
        time_str = f"{int(time_sec // 60):02}:{int(time_sec % 60):02}"
        time_text = self.font_large.render(time_str, True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self._all_cards_matched():
                msg = "YOU WIN!"
            else:
                msg = "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_CURSOR)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "pairs_matched": sum(1 for c in self.cards if c['state'] == 'matched') // 2,
        }

    def _check_termination(self):
        return self.time_remaining <= 0 or self.steps >= 1000 or self._all_cards_matched()
    
    def _all_cards_matched(self):
        return all(card['state'] == 'matched' for card in self.cards)

    def _setup_cards(self):
        self.cards = []
        pattern_ids = list(range(self.PAIR_COUNT)) * 2
        self.np_random.shuffle(pattern_ids)
        
        grid_w = self.SCREEN_WIDTH * 0.8
        grid_h = self.SCREEN_HEIGHT * 0.8
        start_x = (self.SCREEN_WIDTH - grid_w) / 2
        start_y = (self.SCREEN_HEIGHT - grid_h) / 2 + 30
        
        card_w = (grid_w - (self.GRID_COLS - 1) * 10) / self.GRID_COLS
        card_h = (grid_h - (self.GRID_ROWS - 1) * 10) / self.GRID_ROWS
        
        for i in range(self.CARD_COUNT):
            row, col = divmod(i, self.GRID_COLS)
            x = start_x + col * (card_w + 10)
            y = start_y + row * (card_h + 10)
            rect = pygame.Rect(x, y, card_w, card_h)
            
            self.cards.append({
                'rect': rect,
                'pattern_id': pattern_ids[i],
                'state': 'down', # 'down', 'flipping_up', 'up', 'flipping_down', 'matched'
                'flip_progress': 1.0, # 1.0 is fully down or up
            })

    def _draw_cards(self):
        for card in self.cards:
            if card['state'] == 'matched':
                continue # Matched cards are invisible

            rect = card['rect']
            progress = card['flip_progress']
            
            # Animation logic
            if card['state'] in ['flipping_up', 'flipping_down']:
                # Cosine interpolation for smooth flip
                scale = math.cos((progress - 0.5) * math.pi)
                anim_rect = rect.copy()
                anim_rect.width = max(1, int(rect.width * abs(scale)))
                anim_rect.centerx = rect.centerx

                if scale > 0: # Draw back
                    pygame.draw.rect(self.screen, self.COLOR_CARD_DOWN, anim_rect, border_radius=5)
                else: # Draw front
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, anim_rect, border_radius=5)
                    pattern_rect = anim_rect.inflate(-10, -10)
                    if pattern_rect.width > 0 and pattern_rect.height > 0:
                        pattern_surf = self.card_patterns[card['pattern_id']]
                        scaled_pattern = pygame.transform.smoothscale(pattern_surf, pattern_rect.size)
                        self.screen.blit(scaled_pattern, pattern_rect)
            else:
                if card['state'] == 'down':
                    pygame.draw.rect(self.screen, self.COLOR_CARD_DOWN, rect, border_radius=5)
                elif card['state'] == 'up':
                    pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, rect, border_radius=5)
                    pattern_rect = rect.inflate(-10, -10)
                    pattern_surf = self.card_patterns[card['pattern_id']]
                    scaled_pattern = pygame.transform.smoothscale(pattern_surf, pattern_rect.size)
                    self.screen.blit(scaled_pattern, pattern_rect)

    def _draw_cursor(self):
        card_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
        if card_idx < len(self.cards):
            rect = self.cards[card_idx]['rect']
            # Pulsing effect
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
            thickness = 2 + int(pulse * 3)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect.inflate(6, 6), thickness, border_radius=8)

    def _create_particles(self, pos, pattern_id):
        color = self.PATTERN_COLORS[pattern_id]
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(3, 7)
            })

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _pre_render_patterns(self):
        self.card_patterns = []
        size = 100
        draw_funcs = [
            self._draw_pattern_circles, self._draw_pattern_starburst, self._draw_pattern_grid,
            self._draw_pattern_triangles, self._draw_pattern_spiral, self._draw_pattern_stripes,
            self._draw_pattern_cross, self._draw_pattern_dots
        ]
        for i in range(self.PAIR_COUNT):
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            color = self.PATTERN_COLORS[i]
            draw_funcs[i](surf, color)
            self.card_patterns.append(surf)

    # --- Procedural Pattern Drawing Functions ---
    def _draw_pattern_circles(self, surf, color):
        center = (50, 50)
        for r in range(5, 50, 10):
            pygame.gfxdraw.aacircle(surf, center[0], center[1], r, color)
    
    def _draw_pattern_starburst(self, surf, color):
        center = (50, 50)
        for i in range(16):
            angle = (2 * math.pi / 16) * i
            end_x = int(center[0] + 48 * math.cos(angle))
            end_y = int(center[1] + 48 * math.sin(angle))
            pygame.draw.aaline(surf, color, center, (end_x, end_y))

    def _draw_pattern_grid(self, surf, color):
        for i in range(0, 101, 25):
            pygame.draw.line(surf, color, (i, 0), (i, 100), 2)
            pygame.draw.line(surf, color, (0, i), (100, i), 2)

    def _draw_pattern_triangles(self, surf, color):
        pygame.gfxdraw.aapolygon(surf, [(10, 90), (50, 10), (90, 90)], color)
        pygame.gfxdraw.filled_polygon(surf, [(10, 90), (50, 10), (90, 90)], color)

    def _draw_pattern_spiral(self, surf, color):
        center = (50, 50)
        last_pos = center
        for i in range(200):
            angle = 0.1 * i
            r = 0.2 * i
            x = int(center[0] + r * math.cos(angle))
            y = int(center[1] + r * math.sin(angle))
            if (x, y) != last_pos:
                pygame.draw.aaline(surf, color, last_pos, (x, y))
            last_pos = (x, y)

    def _draw_pattern_stripes(self, surf, color):
        for i in range(0, 101, 15):
            pygame.draw.rect(surf, color, (i, 0, 8, 100))

    def _draw_pattern_cross(self, surf, color):
        pygame.draw.rect(surf, color, (40, 10, 20, 80))
        pygame.draw.rect(surf, color, (10, 40, 80, 20))

    def _draw_pattern_dots(self, surf, color):
        for _ in range(50):
            x = self.np_random.integers(10, 90)
            y = self.np_random.integers(10, 90)
            pygame.gfxdraw.aacircle(surf, x, y, 4, color)
            pygame.gfxdraw.filled_circle(surf, x, y, 4, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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