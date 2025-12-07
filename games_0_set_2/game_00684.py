import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to flip a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based memory matching game. Match all pairs before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Pygame Setup (must happen before color definitions) ---
        pygame.init()
        pygame.font.init()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_COLS, self.GRID_ROWS = 8, 4
        self.CARD_COUNT = self.GRID_COLS * self.GRID_ROWS
        self.PAIR_COUNT = self.CARD_COUNT // 2
        self.MAX_TIME = 60.0  # seconds
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 28, 40)
        self.COLOR_GRID = (45, 48, 60)
        self.COLOR_CARD_BACK = (68, 80, 105)
        self.COLOR_CARD_MATCHED = (40, 43, 55)
        self.COLOR_CARD_MATCHED_ICON = (80, 83, 95)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = pygame.Color(0, 255, 127)
        self.COLOR_FAIL = pygame.Color(255, 70, 70)

        # Procedurally generate 16 distinct, vibrant colors for card faces
        self.CARD_FACE_COLORS = [
            pygame.Color(0).lerp(pygame.Color(c), 0.7) for c in
            ['#ff6b6b', '#f9a825', '#f7dc6f', '#69f0ae', '#40e0d0', '#54a0ff', '#9b59b6', '#f5a9b8',
             '#e67e22', '#2ecc71', '#3498db', '#be2edd', '#f1c40f', '#1abc9c', '#d35400', '#8e44ad']
        ]

        # --- Pygame Surfaces & Fonts ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game State (initialized in reset) ---
        self.grid = []
        self.cursor_pos = [0, 0]
        self.flipped_indices = []
        self.mismatched_pair = []
        self.mismatch_timer = 0
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.last_space_held = False
        self.move_cooldown = 0
        self.particles = []
        self.reward_this_step = 0

        # self.reset() # Removed from __init__ to allow external seeding first
        # self.validate_implementation() # Also removed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_TIME
        self.game_over = False
        self.last_space_held = False
        self.move_cooldown = 0
        self.cursor_pos = [0, 0]
        self.flipped_indices = []
        self.mismatched_pair = []
        self.mismatch_timer = 0
        self.particles = []
        self.reward_this_step = 0

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        card_values = list(range(self.PAIR_COUNT)) * 2
        self.np_random.shuffle(card_values)

        self.grid = []
        for i in range(self.CARD_COUNT):
            card = {
                "value": card_values[i],
                "state": "hidden",  # hidden, flipped, matched
                "anim_progress": 0.0,  # 0.0 = hidden, 1.0 = shown
            }
            self.grid.append(card)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1

        self._handle_input(action)
        self._update_game_state()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        reward = self.reward_this_step
        if terminated and not truncated:
            if self.score == self.PAIR_COUNT:  # Win condition
                reward += 100
            elif self.timer <= 0:  # Lose condition
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        if self.move_cooldown == 0 and movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right

            # Wrap around
            self.cursor_pos[0] %= self.GRID_COLS
            self.cursor_pos[1] %= self.GRID_ROWS
            self.move_cooldown = 3  # 3 frames cooldown

        # --- Flip Card ---
        is_space_press = space_held and not self.last_space_held
        if is_space_press and len(self.flipped_indices) < 2:
            idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
            card = self.grid[idx]
            if card["state"] == "hidden" and idx not in self.flipped_indices:
                card["state"] = "flipped"
                self.flipped_indices.append(idx)
                # sfx: card_flip.wav

        self.last_space_held = space_held

    def _update_game_state(self):
        # --- Timer ---
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # --- Mismatch Timer ---
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                for idx in self.mismatched_pair:
                    self.grid[idx]["state"] = "hidden"
                self.mismatched_pair = []

        # --- Card Animations ---
        for card in self.grid:
            target_anim = 1.0 if card["state"] in ["flipped", "matched"] else 0.0
            if card["anim_progress"] != target_anim:
                if card["anim_progress"] < target_anim:
                    card["anim_progress"] = min(target_anim, card["anim_progress"] + 0.15)
                else:
                    card["anim_progress"] = max(target_anim, card["anim_progress"] - 0.15)

        # --- Match Check ---
        if len(self.flipped_indices) == 2 and self.mismatch_timer == 0:
            idx1, idx2 = self.flipped_indices
            card1, card2 = self.grid[idx1], self.grid[idx2]

            if card1["value"] == card2["value"]:
                card1["state"] = "matched"
                card2["state"] = "matched"
                self.score += 1
                self.reward_this_step += 1
                self.flipped_indices = []
                # sfx: match_success.wav

                # Create particles
                pos1 = self._get_card_center(idx1)
                pos2 = self._get_card_center(idx2)
                avg_pos = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                self._create_particles(avg_pos, self.COLOR_SUCCESS, 30)

            else:
                self.mismatch_timer = 15  # frames to show mismatch
                self.mismatched_pair = self.flipped_indices.copy()
                self.flipped_indices = []
                self.reward_this_step -= 0.1
                # sfx: match_fail.wav

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _check_termination(self):
        win = self.score == self.PAIR_COUNT
        lose_time = self.timer <= 0
        
        self.game_over = win or lose_time
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _get_card_pos_and_size(self):
        grid_w = self.WIDTH * 0.8
        grid_h = self.HEIGHT * 0.8
        margin_x = (self.WIDTH - grid_w) / 2
        margin_y = (self.HEIGHT - grid_h) / 2 + 30  # Extra top margin for UI

        card_w = (grid_w - 10 * (self.GRID_COLS + 1)) / self.GRID_COLS
        card_h = (grid_h - 10 * (self.GRID_ROWS + 1)) / self.GRID_ROWS
        return margin_x, margin_y, card_w, card_h

    def _get_card_center(self, index):
        margin_x, margin_y, card_w, card_h = self._get_card_pos_and_size()
        col = index % self.GRID_COLS
        row = index // self.GRID_COLS
        x = margin_x + 10 + col * (card_w + 10)
        y = margin_y + 10 + row * (card_h + 10)
        return (int(x + card_w / 2), int(y + card_h / 2))

    def _render_game(self):
        margin_x, margin_y, card_w, card_h = self._get_card_pos_and_size()

        # --- Render Cards ---
        for i, card in enumerate(self.grid):
            col = i % self.GRID_COLS
            row = i // self.GRID_COLS
            x = margin_x + 10 + col * (card_w + 10)
            y = margin_y + 10 + row * (card_h + 10)

            # Flip animation
            anim_scale = abs(math.cos(card["anim_progress"] * math.pi))
            display_w = int(card_w * anim_scale)

            # Card rect
            card_rect = pygame.Rect(int(x + (card_w - display_w) / 2), int(y), display_w, int(card_h))

            # Draw back or front
            is_showing_front = card["anim_progress"] > 0.5

            if is_showing_front:
                color = self.COLOR_CARD_MATCHED if card["state"] == "matched" else (255, 255, 255)
                pygame.draw.rect(self.screen, color, card_rect, border_radius=5)

                # Draw icon
                icon_color = self.COLOR_CARD_MATCHED_ICON if card["state"] == "matched" else self.CARD_FACE_COLORS[card["value"]]
                icon_radius = int(min(display_w, card_h) * 0.3)
                if icon_radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, card_rect.centerx, card_rect.centery, icon_radius,
                                                  icon_color)
                    pygame.gfxdraw.aacircle(self.screen, card_rect.centerx, card_rect.centery, icon_radius,
                                            icon_color)

            else:  # Showing back
                pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, card_rect, border_radius=5)

            # Mismatch highlight
            if i in self.mismatched_pair and self.mismatch_timer > 0:
                pygame.draw.rect(self.screen, self.COLOR_FAIL, (x, y, card_w, card_h), 3, border_radius=5)

        # --- Render Cursor ---
        cursor_col, cursor_row = self.cursor_pos
        cursor_x = margin_x + 10 + cursor_col * (card_w + 10)
        cursor_y = margin_y + 10 + cursor_row * (card_h + 10)
        cursor_rect = pygame.Rect(int(cursor_x - 4), int(cursor_y - 4), int(card_w + 8), int(card_h + 8))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

        # --- Render Particles ---
        for p in self.particles:
            size = int(p['life'] / p['max_life'] * 5)
            if size > 1:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # --- Timer Bar ---
        timer_ratio = self.timer / self.MAX_TIME
        bar_width = self.WIDTH * 0.5
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10

        timer_color = self.COLOR_SUCCESS.lerp(self.COLOR_FAIL, 1 - timer_ratio)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, bar_width * timer_ratio, bar_height), border_radius=5)

        # --- Score Text ---
        score_text = self.font_main.render(f"PAIRS: {self.score}/{self.PAIR_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))

            if self.score == self.PAIR_COUNT:
                msg = "YOU WIN!"
                color = self.COLOR_SUCCESS
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_FAIL

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color
            })

    def render(self):
        return self._get_observation()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Switch to a visible video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset(seed=42)

    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Memory Match")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Terminated: {terminated}, Truncated: {truncated}")
            pygame.time.wait(2000)  # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(env.FPS)

    pygame.quit()