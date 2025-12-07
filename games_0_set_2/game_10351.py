import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:20:04.071372
# Source Brief: brief_00351.md
# Brief Index: 351
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Predict the opponent's pitch based on a quick visual tell. "
        "Guess correctly to score points and win the round."
    )
    user_guide = (
        "Controls: Use ↑, ↓, ←, → to select one of the first four pitches. "
        "Press space to select the fifth pitch."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Action and Observation Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode  # Not used, but good practice

        # --- Visual & Game Design Constants ---
        self.PITCH_TYPES = ["Fastball", "Curveball", "Slider", "Changeup", "Sinker"]
        self.PITCH_COLORS = [
            (255, 80, 80),  # Red
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (80, 255, 80),  # Green
            (255, 120, 255),  # Magenta
        ]
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_FIELD = (35, 60, 45)
        self.COLOR_LINES = (200, 200, 180)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_OPPONENT = (255, 140, 60)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        self.FONT_LARGE = pygame.font.Font(None, 64)
        self.FONT_MEDIUM = pygame.font.Font(None, 36)
        self.FONT_SMALL = pygame.font.Font(None, 24)

        self.MAX_STEPS = 3000  # Generous limit, approx 100 seconds
        self.ROUNDS_TO_WIN = 3
        self.PITCHES_PER_ROUND = 3
        self.SELECT_TIME_LIMIT = 3.0  # seconds

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0  # Cumulative reward
        self.game_over = False
        self.player_rounds_won = 0
        self.opponent_rounds_won = 0
        self.round_number = 0
        self.pitch_in_round = 0
        self.player_pitch_points = 0
        self.opponent_pitch_points = 0

        self.player_selection = -1  # 0-4 for pitch, -1 for none
        self.opponent_selection = -1

        self.phase = "IDLE"  # TELL, SELECT, REVEAL, OUTCOME, ROUND_END, GAME_OVER
        self.phase_timer = 0.0

        self.tell_duration = 1.0

        self.pending_reward = 0.0
        self.last_action_time = 0.0

        # Animation state
        self.reveal_anim_progress = 0.0
        self.outcome_anim_progress = 0.0
        self.last_outcome_correct = False

        self.np_random = None # Will be seeded in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_rounds_won = 0
        self.opponent_rounds_won = 0
        self.round_number = 0
        self.tell_duration = 1.0  # Reset difficulty
        self.pending_reward = 0.0

        self._start_new_round()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- Time & Step Management ---
        dt = self.clock.tick(self.metadata["render_fps"]) / 1000.0
        self.steps += 1
        self.phase_timer = max(0, self.phase_timer - dt)

        reward = 0.0

        if not self.game_over:
            # --- Phase-based Game Logic ---
            if self.phase == "TELL":
                if self.phase_timer <= 0:
                    self.phase = "SELECT"
                    self.phase_timer = self.SELECT_TIME_LIMIT
                    self.player_selection = -1
                    self.last_action_time = 0.0

            elif self.phase == "SELECT":
                self._process_player_input(action)
                self.last_action_time += dt

                # Player made a choice or timer ran out
                if self.player_selection != -1 or self.phase_timer <= 0:
                    if self.player_selection == -1:
                        # Timer ran out, random choice (no-op penalty)
                        self.player_selection = self.np_random.integers(0, 5)
                        # SFX: time_out_sound
                    self._resolve_pitch()
                    reward += self.pending_reward
                    self.pending_reward = 0.0
                    self.phase = "REVEAL"
                    self.phase_timer = 1.0  # Reveal animation duration
                    self.reveal_anim_progress = 0.0
                    # SFX: card_reveal_swoosh

            elif self.phase == "REVEAL":
                self.reveal_anim_progress = min(
                    1.0, self.reveal_anim_progress + dt / 0.5
                )
                if self.phase_timer <= 0:
                    self.phase = "OUTCOME"
                    self.phase_timer = 1.5  # Outcome display duration
                    self.outcome_anim_progress = 0.0

            elif self.phase == "OUTCOME":
                self.outcome_anim_progress = min(
                    1.0, self.outcome_anim_progress + dt / 0.5
                )
                if self.phase_timer <= 0:
                    if self.pitch_in_round >= self.PITCHES_PER_ROUND:
                        self._resolve_round()
                        reward += self.pending_reward
                        self.pending_reward = 0.0
                        if (
                            self.player_rounds_won >= self.ROUNDS_TO_WIN
                            or self.opponent_rounds_won >= self.ROUNDS_TO_WIN
                        ):
                            self.phase = "GAME_OVER"
                            self.game_over = True
                            self._resolve_game()
                            reward += self.pending_reward
                            self.pending_reward = 0.0
                        else:
                            self.phase = "ROUND_END"
                            self.phase_timer = 2.0
                    else:
                        self._start_new_pitch()

            elif self.phase == "ROUND_END":
                if self.phase_timer <= 0:
                    self._start_new_round()

        self.score += reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _process_player_input(self, action):
        movement, space_held, _ = action

        # Priority: Space > Movement. Only register one selection per pitch.
        if self.player_selection == -1:
            if space_held == 1:
                self.player_selection = 4  # Card 5 is index 4
                # SFX: select_confirm_sound
            elif movement > 0:
                self.player_selection = movement - 1  # Cards 1-4 are indices 0-3
                # SFX: select_confirm_sound

    def _start_new_round(self):
        self.round_number += 1
        self.player_pitch_points = 0
        self.opponent_pitch_points = 0
        self._start_new_pitch()

    def _start_new_pitch(self):
        self.pitch_in_round = (
            self.player_pitch_points + self.opponent_pitch_points + 1
        )
        self.opponent_selection = self.np_random.integers(0, 5)
        self.player_selection = -1
        self.phase = "TELL"
        self.phase_timer = self.tell_duration
        self.reveal_anim_progress = 0.0
        self.outcome_anim_progress = 0.0
        # SFX: new_pitch_alert_sound

    def _resolve_pitch(self):
        if self.player_selection == self.opponent_selection:
            self.player_pitch_points += 1
            self.last_outcome_correct = True
            # Fast prediction reward
            if self.last_action_time < 1.0:
                self.pending_reward = 1.0
            else:
                self.pending_reward = 0.0  # Slow correct prediction
            # SFX: success_chime
        else:
            self.opponent_pitch_points += 1
            self.last_outcome_correct = False
            self.pending_reward = -1.0
            # SFX: failure_buzz

    def _resolve_round(self):
        if self.player_pitch_points > self.opponent_pitch_points:
            self.player_rounds_won += 1
            self.pending_reward = 5.0
            # Increase difficulty
            self.tell_duration = max(0.2, self.tell_duration - 0.05)
            # SFX: round_win_fanfare
        else:
            self.opponent_rounds_won += 1
            self.pending_reward = -5.0
            # SFX: round_lose_sound

    def _resolve_game(self):
        if self.player_rounds_won >= self.ROUNDS_TO_WIN:
            self.pending_reward = 50.0
            # SFX: game_win_jingle
        else:
            self.pending_reward = -50.0
            # SFX: game_lose_trombone

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw baseball diamond
        field_rect = pygame.Rect(50, 100, self.WIDTH - 100, self.HEIGHT - 120)
        pygame.draw.rect(self.screen, self.COLOR_FIELD, field_rect, border_radius=10)

        home_plate = (field_rect.centerx, field_rect.bottom - 20)
        first_base = (field_rect.right - 20, field_rect.centery)
        second_base = (field_rect.centerx, field_rect.top + 20)
        third_base = (field_rect.left + 20, field_rect.centery)

        pygame.draw.line(self.screen, self.COLOR_LINES, home_plate, first_base, 3)
        pygame.draw.line(self.screen, self.COLOR_LINES, first_base, second_base, 3)
        pygame.draw.line(self.screen, self.COLOR_LINES, second_base, third_base, 3)
        pygame.draw.line(self.screen, self.COLOR_LINES, third_base, home_plate, 3)

        # Opponent Tell
        if self.phase == "TELL" and self.opponent_selection != -1:
            tell_color = self.PITCH_COLORS[self.opponent_selection]
            # Pulsing glow effect
            alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.01)
            pygame.gfxdraw.filled_circle(
                self.screen,
                field_rect.centerx,
                second_base[1] + 30,
                15,
                (*tell_color, int(alpha)),
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                field_rect.centerx,
                second_base[1] + 30,
                15,
                (*tell_color, int(alpha)),
            )

        # Cards
        card_width, card_height = 80, 50
        card_y_player = self.HEIGHT - 60
        card_y_opponent = 40
        total_card_width = 5 * card_width + 4 * 10
        start_x = (self.WIDTH - total_card_width) // 2

        for i in range(5):
            card_x = start_x + i * (card_width + 10)

            # Player cards
            is_selected = self.player_selection == i
            self._draw_card(
                card_x, card_y_player, card_width, card_height, i, self.COLOR_PLAYER, is_selected
            )

            # Opponent cards
            if self.phase in ["REVEAL", "OUTCOME"]:
                self._draw_card(
                    card_x,
                    card_y_opponent,
                    card_width,
                    card_height,
                    i,
                    self.COLOR_OPPONENT,
                    i == self.opponent_selection,
                    reveal_progress=self.reveal_anim_progress,
                )
            else:
                self._draw_card(
                    card_x,
                    card_y_opponent,
                    card_width,
                    card_height,
                    i,
                    self.COLOR_OPPONENT,
                    False,
                    is_hidden=True,
                )

        # Outcome animation
        if self.phase == "OUTCOME":
            size = int(100 * self.outcome_anim_progress)
            center = (self.WIDTH // 2, self.HEIGHT // 2)
            if self.last_outcome_correct:
                color = self.COLOR_SUCCESS
                # Draw a checkmark
                p1 = (center[0] - size // 2, center[1])
                p2 = (center[0] - size // 4, center[1] + size // 2)
                p3 = (center[0] + size // 2, center[1] - size // 2)
                pygame.draw.line(self.screen, color, p1, p2, 15)
                pygame.draw.line(self.screen, color, p2, p3, 15)
            else:
                color = self.COLOR_FAIL
                # Draw an X
                p1 = (center[0] - size // 2, center[1] - size // 2)
                p2 = (center[0] + size // 2, center[1] + size // 2)
                p3 = (center[0] + size // 2, center[1] - size // 2)
                p4 = (center[0] - size // 2, center[1] + size // 2)
                pygame.draw.line(self.screen, color, p1, p2, 15)
                pygame.draw.line(self.screen, color, p3, p4, 15)

    def _draw_card(
        self, x, y, w, h, pitch_idx, base_color, is_selected, is_hidden=False, reveal_progress=0.0
    ):
        final_w, final_h = w, h

        if is_selected and reveal_progress > 0:  # Opponent's revealed card
            scale = 1.0 + 0.5 * math.sin(reveal_progress * math.pi)  # Pop effect
            w, h = int(w * scale), int(h * scale)
            x, y = x - (w - final_w) // 2, y - (h - final_h) // 2

        rect = pygame.Rect(x, y, w, h)

        if is_hidden:
            pygame.draw.rect(self.screen, (80, 80, 90), rect, border_radius=5)
            return

        color = self.PITCH_COLORS[pitch_idx]
        pygame.draw.rect(self.screen, color, rect, border_radius=5)

        if is_selected and reveal_progress == 0:  # Player's selected card
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 4, border_radius=5)

        text = self.PITCH_TYPES[pitch_idx][:4]
        self._draw_text(text, self.FONT_SMALL, (x + w // 2, y + h // 2), (0, 0, 0))

    def _render_ui(self):
        # Score display
        self._draw_text(f"Player: {self.player_rounds_won}", self.FONT_MEDIUM, (80, 20))
        self._draw_text(
            f"Opponent: {self.opponent_rounds_won}",
            self.FONT_MEDIUM,
            (self.WIDTH - 100, 20),
        )

        # Round and Pitch info
        info_text = (
            f"Round {self.round_number} - Pitch {self.pitch_in_round}/{self.PITCHES_PER_ROUND}"
        )
        self._draw_text(info_text, self.FONT_MEDIUM, (self.WIDTH // 2, 20))

        # Selection Timer Bar
        if self.phase == "SELECT":
            bar_width = 200
            bar_height = 20
            bar_x = (self.WIDTH - bar_width) // 2
            bar_y = self.HEIGHT - 95
            fill_ratio = self.phase_timer / self.SELECT_TIME_LIMIT

            pygame.draw.rect(
                self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=5
            )
            pygame.draw.rect(
                self.screen,
                self.COLOR_PLAYER,
                (bar_x, bar_y, int(bar_width * fill_ratio), bar_height),
                border_radius=5,
            )

        # Phase/Game Over Text
        if self.phase == "GAME_OVER":
            if self.player_rounds_won >= self.ROUNDS_TO_WIN:
                self._draw_text(
                    "YOU WIN!",
                    self.FONT_LARGE,
                    (self.WIDTH // 2, self.HEIGHT // 2),
                    self.COLOR_SUCCESS,
                )
            else:
                self._draw_text(
                    "GAME OVER",
                    self.FONT_LARGE,
                    (self.WIDTH // 2, self.HEIGHT // 2),
                    self.COLOR_FAIL,
                )
        elif self.phase == "ROUND_END":
            if self.player_pitch_points > self.opponent_pitch_points:
                self._draw_text(
                    "ROUND WON",
                    self.FONT_LARGE,
                    (self.WIDTH // 2, self.HEIGHT // 2),
                    self.COLOR_SUCCESS,
                )
            else:
                self._draw_text(
                    "ROUND LOST",
                    self.FONT_LARGE,
                    (self.WIDTH // 2, self.HEIGHT // 2),
                    self.COLOR_FAIL,
                )

    def _draw_text(self, text, font, center_pos, color=None, shadow_color=None):
        if color is None:
            color = self.COLOR_TEXT
        if shadow_color is None:
            shadow_color = self.COLOR_TEXT_SHADOW

        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)

        shadow_surf = font.render(text, True, shadow_color)
        shadow_rect = shadow_surf.get_rect(center=(center_pos[0] + 2, center_pos[1] + 2))

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_rounds_won": self.player_rounds_won,
            "opponent_rounds_won": self.opponent_rounds_won,
            "round_number": self.round_number,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Mapping keyboard keys to MultiDiscrete actions
    #   - Arrows for cards 1-4
    #   - Space for card 5
    key_to_action = {
        pygame.K_UP: [1, 0, 0],
        pygame.K_DOWN: [2, 0, 0],
        pygame.K_LEFT: [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
    }

    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Baseball Prediction")
    game_clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print("Up/Down/Left/Right: Select pitches 1-4")
    print("Spacebar: Select pitch 5")
    print("R: Reset environment")
    print("Q: Quit")

    while not done:
        action = [0, 0, 0]  # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key in key_to_action:
                    action = key_to_action[event.key]

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(
                f"Step: {info['steps']}, Reward: {reward:.2f}, Total Score: {info['score']:.2f}"
            )

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        game_clock.tick(env.metadata["render_fps"])

    env.close()