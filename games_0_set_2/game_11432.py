import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:58:18.524308
# Source Brief: brief_01432.md
# Brief Index: 1432
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Type the words falling from the sky before they reach the bottom. "
        "Build your word one character at a time and submit it to score points."
    )
    user_guide = (
        "Controls: Use ↑↓ to cycle letters and ←→ to move the cursor. "
        "Press space to add a new letter and shift to submit the current word."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 10000
    MAX_MISSES = 5

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WORD = (255, 255, 255)
    COLOR_TYPED = (100, 255, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_WARNING_LINE = (200, 0, 0)

    WARNING_LINE_Y = 360

    # Word list (no external files)
    WORD_LIST = [
        "python", "agent", "learn", "reward", "action", "state", "game", "code",
        "terminal", "vector", "pixel", "window", "event", "render", "step", "reset",
        "expert", "visual", "quality", "brief", "design", "system", "space", "model",
        "future", "past", "value", "policy", "network", "deep", "neural", "train",
        "loop", "logic", "bug", "test", "final", "play", "feel", "fun", "type",
        "word", "fast", "slow", "score", "high", "level", "data", "array", "numpy",
        "float", "integer", "error", "check", "rule", "goal", "node", "tree", "graph"
    ]

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_word = pygame.font.SysFont('consolas', 24, bold=True)
        self.font_ui = pygame.font.SysFont('dejavusansmono', 20, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.words = None
        self.missed_words = None
        self.correctly_typed_words = None
        self.word_speed = None
        self.typed_string = None
        self.cursor_pos = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None
        self.time_elapsed = None
        self.total_typed_chars_in_session = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.words = []
        self.missed_words = 0
        self.correctly_typed_words = 0
        self.word_speed = 1.0

        self.typed_string = ""
        self.cursor_pos = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.time_elapsed = 0.0
        self.total_typed_chars_in_session = 0

        # Spawn initial words
        for _ in range(3):
            self._spawn_word()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed += 1.0 / self.FPS

        reward = 0

        # --- Handle Actions ---
        reward += self._handle_actions(action)

        # --- Update Game State ---
        self._update_words()
        self._update_particles()

        # --- Calculate Score (WPM) ---
        if self.time_elapsed > 1:  # Avoid division by zero and unstable early values
            minutes = self.time_elapsed / 60.0
            wpm = (self.total_typed_chars_in_session / 5.0) / minutes
            self.score = int(wpm)
        else:
            self.score = 0

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.missed_words >= self.MAX_MISSES:
                reward = -100  # Failure penalty
            elif truncated:
                reward += 10 # Survival bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Action [0]: Movement (Character/Cursor Manipulation) ---
        if self.typed_string and 0 <= self.cursor_pos < len(self.typed_string):
            char_list = list(self.typed_string)
            current_char = char_list[self.cursor_pos]
            current_idx = self.ALPHABET.find(current_char)

            if movement == 1:  # Up: cycle character forward
                if current_idx != -1:
                    char_list[self.cursor_pos] = self.ALPHABET[(current_idx + 1) % len(self.ALPHABET)]
                    self.typed_string = "".join(char_list)
            elif movement == 2:  # Down: cycle character backward
                if current_idx != -1:
                    char_list[self.cursor_pos] = self.ALPHABET[(current_idx - 1 + len(self.ALPHABET)) % len(self.ALPHABET)]
                    self.typed_string = "".join(char_list)

        if movement == 3:  # Left: move cursor left
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif movement == 4:  # Right: move cursor right
            self.cursor_pos = min(len(self.typed_string), self.cursor_pos + 1)

        # --- Action [1]: Space (Add new character) ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # Insert 'a' at the cursor position
            self.typed_string = self.typed_string[:self.cursor_pos] + 'a' + self.typed_string[self.cursor_pos:]
            self.cursor_pos += 1
            # # sound: key_press_soft

        # --- Action [2]: Shift (Submit word) ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.typed_string:
            found_match = False
            for i, word_obj in enumerate(self.words):
                if self.typed_string == word_obj['text']:
                    # --- Correct Word ---
                    reward += 10
                    self.total_typed_chars_in_session += len(word_obj['text'])
                    self.correctly_typed_words += 1

                    # Create particle explosion
                    self._create_particles(word_obj['pos'], self.COLOR_TYPED)

                    # Remove word and spawn more
                    del self.words[i]
                    self._spawn_word()
                    self._spawn_word()
                    # # sound: word_complete

                    # Increase difficulty every 10 words
                    if self.correctly_typed_words > 0 and self.correctly_typed_words % 10 == 0:
                        self.word_speed = min(5.0, self.word_speed + 0.2)
                        # # sound: level_up

                    found_match = True
                    break  # Stop after finding one match

            if not found_match:
                reward -= 1  # Small penalty for wrong submission
                # # sound: error

            # Reset typed string
            self.typed_string = ""
            self.cursor_pos = 0

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return reward

    def _update_words(self):
        words_to_remove = []
        for i, word in enumerate(self.words):
            word['pos'][1] += word['speed']
            if word['pos'][1] > self.WARNING_LINE_Y:
                words_to_remove.append(i)
                self.missed_words += 1
                self._create_particles(word['pos'], self.COLOR_WARNING_LINE)
                self._spawn_word()  # Replace missed word
                # # sound: word_missed

        # Remove words from list in reverse order to avoid index errors
        for i in sorted(words_to_remove, reverse=True):
            del self.words[i]

    def _spawn_word(self):
        text = self.np_random.choice(self.WORD_LIST)
        text_width, _ = self.font_word.size(text)

        # Attempt to find a non-overlapping horizontal position
        max_attempts = 10
        x = 0
        for _ in range(max_attempts):
            x = self.np_random.integers(10, self.SCREEN_WIDTH - text_width - 10)
            new_rect = pygame.Rect(x, -20, text_width, 30)
            is_overlapping = False
            for other_word in self.words:
                other_text_width, _ = self.font_word.size(other_word['text'])
                other_rect = pygame.Rect(other_word['pos'][0], other_word['pos'][1], other_text_width, 30)
                if new_rect.colliderect(other_rect):
                    is_overlapping = True
                    break
            if not is_overlapping:
                break
        else:  # If all attempts failed, just place it randomly
            x = self.np_random.integers(10, self.SCREEN_WIDTH - text_width - 10)

        self.words.append({
            'text': text,
            'pos': [x, 0],
            'speed': self.word_speed * self.np_random.uniform(0.9, 1.1)
        })

    def _check_termination(self):
        return self.missed_words >= self.MAX_MISSES

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
            "missed_words": self.missed_words,
            "correct_words": self.correctly_typed_words,
            "word_speed": self.word_speed
        }

    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color']
            )

        # Render falling words
        for word in self.words:
            text_surface = self.font_word.render(word['text'], True, self.COLOR_WORD)
            self.screen.blit(text_surface, (int(word['pos'][0]), int(word['pos'][1])))

    def _render_ui(self):
        # Render warning line
        pygame.draw.line(self.screen, self.COLOR_WARNING_LINE, (0, self.WARNING_LINE_Y),
                         (self.SCREEN_WIDTH, self.WARNING_LINE_Y), 2)

        # Render typed string
        typed_surface = self.font_word.render(self.typed_string, True, self.COLOR_TYPED)
        typed_pos_x = (self.SCREEN_WIDTH - typed_surface.get_width()) // 2
        typed_pos_y = self.SCREEN_HEIGHT - 30
        self.screen.blit(typed_surface, (typed_pos_x, typed_pos_y))

        # Render cursor
        if self.steps % self.FPS < self.FPS / 2:  # Blinking cursor
            cursor_render_pos = self.cursor_pos
            pre_cursor_text = self.typed_string[:cursor_render_pos]
            pre_cursor_width, _ = self.font_word.size(pre_cursor_text)
            cursor_x = typed_pos_x + pre_cursor_width
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, typed_pos_y), (cursor_x, typed_pos_y + 24), 2)

        # Render WPM
        wpm_text = f"WPM: {self.score}"
        wpm_surface = self.font_ui.render(wpm_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(wpm_surface, (10, 10))

        # Render Misses
        misses_text = f"MISSES: {self.missed_words}/{self.MAX_MISSES}"
        misses_surface = self.font_ui.render(misses_text, True, self.COLOR_UI_TEXT)
        misses_rect = misses_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(misses_surface, misses_rect)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
            if p['life'] <= 0 or p['radius'] <= 0:
                particles_to_remove.append(i)

        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Typing Avalanche")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False

    # --- Manual Control Mapping ---
    # Arrow keys -> Movement (for cursor/char cycle)
    # Space -> Add char 'a'
    # Left Shift -> Submit word

    while not terminated and not truncated:
        movement = 0  # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        # Use a simple mechanism to handle key presses vs holds for movement
        # This is not perfect but works for manual play
        if any(keys):
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")

    env.close()