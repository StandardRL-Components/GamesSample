
# Generated: 2025-08-27T19:40:12.008077
# Source Brief: brief_02218.md
# Brief Index: 2218

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Memory Match game. Find all pairs before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_CARD_HIDDEN = (60, 80, 100)
        self.COLOR_CARD_MATCHED = (45, 60, 75)
        self.COLOR_CURSOR = (255, 180, 0)
        self.COLOR_INCORRECT = (255, 80, 80)
        self.COLOR_CORRECT = (0, 255, 120)
        self.SHAPE_COLORS = [
            (255, 87, 34),   # Deep Orange
            (33, 150, 243),  # Blue
            (76, 175, 80),   # Green
            (255, 235, 59),  # Yellow
            (156, 39, 176),  # Purple
            (0, 188, 212),   # Cyan
            (233, 30, 99),   # Pink
            (255, 255, 255), # White
        ]

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.GRID_SIZE = (4, 4)
        self.NUM_PAIRS = (self.GRID_SIZE[0] * self.GRID_SIZE[1]) // 2
        self.INITIAL_TIME = 60 * self.FPS # 60 seconds
        self.MISMATCH_DELAY = 1 * self.FPS # 1 second
        self.MOVE_COOLDOWN_FRAMES = 4

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.cards = []
        self.cursor_pos = [0, 0]
        self.selection_state = "AWAITING_FIRST_PICK"
        self.first_selection_idx = None
        self.second_selection_idx = None
        self.mismatch_timer = 0
        self.last_space_state = False
        self.move_cooldown = 0
        self.particles = []
        self.matched_pairs = 0

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.INITIAL_TIME
        self.cursor_pos = [0, 0]
        self.selection_state = "AWAITING_FIRST_PICK"
        self.first_selection_idx = None
        self.second_selection_idx = None
        self.mismatch_timer = 0
        self.last_space_state = False
        self.move_cooldown = 0
        self.particles = []
        self.matched_pairs = 0

        # Create and shuffle cards
        card_values = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(card_values)

        self.cards = []
        for i, value in enumerate(card_values):
            self.cards.append({
                'value': value,
                'state': 'hidden', # hidden, revealed, matched
                'anim_progress': 1.0, # 0.0 to 1.0
                'anim_state': 'idle', # idle, shrinking, growing
            })

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        current_reward = self._handle_input(action)
        self._update_game_state()
        
        # Decrement timer and update steps
        self.time_left = max(0, self.time_left - 1)
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.matched_pairs == self.NUM_PAIRS:
                current_reward += 50 # Win bonus
            else:
                current_reward -= 100 # Timeout penalty
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            current_reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if self.move_cooldown == 0 and movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE[0]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE[1]
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        # --- Card Selection (on space press, not hold) ---
        space_just_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held

        if space_just_pressed:
            cursor_idx = self.cursor_pos[1] * self.GRID_SIZE[1] + self.cursor_pos[0]
            selected_card = self.cards[cursor_idx]

            # Can only select hidden cards not currently being revealed
            if selected_card['state'] == 'hidden' and selected_card['anim_state'] == 'idle':
                if self.selection_state == "AWAITING_FIRST_PICK":
                    selected_card['state'] = 'revealed'
                    selected_card['anim_state'] = 'shrinking'
                    selected_card['anim_progress'] = 0.0
                    self.first_selection_idx = cursor_idx
                    self.selection_state = "AWAITING_SECOND_PICK"
                    reward += 0.1 # Reward for revealing a new card
                    # sfx: card_flip_1
                elif self.selection_state == "AWAITING_SECOND_PICK" and cursor_idx != self.first_selection_idx:
                    selected_card['state'] = 'revealed'
                    selected_card['anim_state'] = 'shrinking'
                    selected_card['anim_progress'] = 0.0
                    self.second_selection_idx = cursor_idx
                    self.selection_state = "CHECKING_MATCH"
                    reward += 0.1 # Reward for revealing a new card
                    # sfx: card_flip_2
            else:
                reward -= 0.1 # Penalty for selecting non-hidden/animating card

        return reward

    def _update_game_state(self):
        # --- Card Animations ---
        for card in self.cards:
            if card['anim_state'] != 'idle':
                card['anim_progress'] = min(1.0, card['anim_progress'] + 1.0 / (self.FPS * 0.15)) # 0.15s flip
                if card['anim_progress'] >= 1.0:
                    if card['anim_state'] == 'shrinking':
                        card['anim_state'] = 'growing'
                        card['anim_progress'] = 0.0
                    elif card['anim_state'] == 'growing':
                        card['anim_state'] = 'idle'
                        card['anim_progress'] = 1.0

        # --- Match Logic ---
        if self.selection_state == "CHECKING_MATCH":
            card1 = self.cards[self.first_selection_idx]
            card2 = self.cards[self.second_selection_idx]
            
            # Wait for both cards to finish flipping before checking
            if card1['anim_state'] == 'idle' and card2['anim_state'] == 'idle':
                if card1['value'] == card2['value']:
                    # Match!
                    card1['state'] = 'matched'
                    card2['state'] = 'matched'
                    self.score += 10
                    self.matched_pairs += 1
                    self._create_particles(self.first_selection_idx, self.second_selection_idx, self.SHAPE_COLORS[card1['value']])
                    self.selection_state = "AWAITING_FIRST_PICK"
                    self.first_selection_idx, self.second_selection_idx = None, None
                    # sfx: match_success
                else:
                    # Mismatch
                    self.selection_state = "DISPLAYING_MISMATCH"
                    self.mismatch_timer = self.MISMATCH_DELAY
                    # sfx: match_fail

        # --- Mismatch Timer ---
        if self.selection_state == "DISPLAYING_MISMATCH":
            self.mismatch_timer -= 1
            if self.mismatch_timer <= 0:
                card1, card2 = self.cards[self.first_selection_idx], self.cards[self.second_selection_idx]
                card1['state'], card2['state'] = 'hidden', 'hidden'
                card1['anim_state'], card2['anim_state'] = 'shrinking', 'shrinking'
                card1['anim_progress'], card2['anim_progress'] = 0.0, 0.0
                self.selection_state = "AWAITING_FIRST_PICK"
                self.first_selection_idx, self.second_selection_idx = None, None

        # --- Particle Physics ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return (
            self.time_left <= 0
            or self.steps >= self.MAX_STEPS
            or self.matched_pairs == self.NUM_PAIRS
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
            "time_left_seconds": self.time_left / self.FPS,
            "matched_pairs": self.matched_pairs,
        }

    def _get_card_rect(self, index):
        rows, cols = self.GRID_SIZE
        grid_w, grid_h = 560, 320
        margin_x = (self.width - grid_w) / 2
        margin_y = (self.height - grid_h) / 2 + 40 # Offset for UI
        card_w, card_h = grid_w / cols, grid_h / rows
        padding = 10
        row, col = index // cols, index % cols
        x = margin_x + col * card_w + padding / 2
        y = margin_y + row * card_h + padding / 2
        return pygame.Rect(x, y, card_w - padding, card_h - padding)

    def _render_game(self):
        self._render_cards()
        self._render_particles()
        self._render_cursor()

    def _render_cards(self):
        for i, card in enumerate(self.cards):
            rect = self._get_card_rect(i)
            
            color = self.COLOR_CARD_HIDDEN
            if card['state'] == 'matched':
                color = self.COLOR_CARD_MATCHED
            elif self.selection_state == "DISPLAYING_MISMATCH" and (i == self.first_selection_idx or i == self.second_selection_idx):
                color = self.COLOR_INCORRECT
            elif card['state'] == 'revealed':
                color = self.COLOR_GRID

            anim_width = rect.width
            if card['anim_state'] == 'shrinking':
                anim_width = rect.width * (1.0 - card['anim_progress'])
            elif card['anim_state'] == 'growing':
                anim_width = rect.width * card['anim_progress']

            anim_rect = pygame.Rect(rect.centerx - anim_width / 2, rect.y, anim_width, rect.height)
            pygame.draw.rect(self.screen, color, anim_rect, border_radius=8)

            if card['state'] != 'hidden' and card['anim_state'] != 'shrinking':
                shape_color = self.SHAPE_COLORS[card['value']]
                scale = 1.0
                if card['anim_state'] == 'growing' and card['anim_progress'] > 0.5:
                    scale = (card['anim_progress'] - 0.5) * 2
                self._draw_shape(rect.center, card['value'], shape_color, scale)

    def _draw_shape(self, center, value, color, scale=1.0):
        if scale <= 0: return
        size = 25 * scale
        cx, cy = int(center[0]), int(center[1])

        if value == 0: pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(size), color)
        elif value == 1: pygame.draw.rect(self.screen, color, (cx - size, cy - size, 2 * size, 2 * size))
        elif value == 2: pygame.gfxdraw.filled_polygon(self.screen, [(cx, cy - size * 1.15), (cx - size, cy + size * 0.58), (cx + size, cy + size * 0.58)], color)
        elif value == 3: pygame.gfxdraw.filled_polygon(self.screen, [(cx, cy - size), (cx - size, cy), (cx, cy + size), (cx + size, cy)], color)
        elif value == 4:
            points = []
            for i in range(10):
                r = size if i % 2 == 0 else size * 0.4
                angle = i * math.pi / 5 - math.pi / 2
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif value == 5:
            points = [(cx + size * math.cos(i * math.pi / 3), cy + size * math.sin(i * math.pi / 3)) for i in range(6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif value == 6:
            thick = int(size * 0.3)
            pygame.draw.rect(self.screen, color, (cx - size, cy - thick, 2 * size, 2 * thick))
            pygame.draw.rect(self.screen, color, (cx - thick, cy - size, 2 * thick, 2 * size))
        elif value == 7:
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(size), color)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(size * 0.6), self.COLOR_GRID)

    def _render_cursor(self):
        cursor_idx = self.cursor_pos[1] * self.GRID_SIZE[1] + self.cursor_pos[0]
        rect = self._get_card_rect(cursor_idx)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        thickness = 3 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, thickness, border_radius=10)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(p['radius'] * life_ratio)
            if radius > 0:
                color = p['color'] + (int(255 * life_ratio),)
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (p['pos'][0] - radius, p['pos'][1] - radius))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        time_seconds = math.ceil(self.time_left / self.FPS)
        timer_color = self.COLOR_TEXT if time_seconds > 10 else self.COLOR_INCORRECT
        timer_text = self.font_large.render(f"TIME: {time_seconds}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.width - 20, 10))
        self.screen.blit(timer_text, timer_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.matched_pairs == self.NUM_PAIRS else "TIME'S UP!"
            win_text = self.font_large.render(msg, True, self.COLOR_CORRECT if self.matched_pairs == self.NUM_PAIRS else self.COLOR_INCORRECT)
            win_rect = win_text.get_rect(center=(self.width / 2, self.height / 2 - 20))
            self.screen.blit(win_text, win_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.width / 2, self.height / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _create_particles(self, idx1, idx2, color):
        rect1, rect2 = self._get_card_rect(idx1), self._get_card_rect(idx2)
        center_pos = [(rect1.centerx + rect2.centerx) / 2, (rect1.centery + rect2.centery) / 2]
        for _ in range(30):
            angle, speed = random.uniform(0, 2 * math.pi), random.uniform(1, 4)
            life = random.randint(int(self.FPS * 0.5), int(self.FPS * 1.0))
            self.particles.append({
                'pos': list(center_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.randint(5, 12),
                'life': life, 'max_life': life, 'color': color,
            })
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To play manually, ensure you have pygame installed (`pip install pygame`)
    # and run this file directly.
    # For headless execution (e.g., in a cloud environment), you might need to
    # set an environment variable:
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    try:
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Memory Match")
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + "="*30)
        print("MANUAL PLAYBACK")
        print(env.user_guide)
        print("Press 'R' to reset.")
        print("="*30 + "\n")
        
        while True:
            # --- Human Input to Action ---
            movement, space, shift = 0, 0, 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            action = [movement, space, shift]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Rendering ---
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Clock ---
            should_quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_quit = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
            if should_quit:
                break
            
            env.clock.tick(env.FPS)
            
    finally:
        env.close()
        print("Environment closed.")