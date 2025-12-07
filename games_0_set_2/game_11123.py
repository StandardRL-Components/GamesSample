import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:40:00.888331
# Source Brief: brief_01123.md
# Brief Index: 1123
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Wordfall Rhapsody: A rhythm-based typing game Gymnasium environment.

    The agent controls a cursor on a virtual keyboard to type falling words.
    Points are awarded for correct words, with a multiplier for submissions
    timed to the beat of a rhythmic pulse. The goal is to reach 1000 points
    within the time limit.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0:none, 1:up, 2:down, 3:left, 4:right) - Navigates the virtual keyboard.
    - action[1]: Space (0:released, 1:held) - "Presses" the selected key on the virtual keyboard.
    - action[2]: Shift (0:released, 1:held) - Toggles the case of the virtual keyboard.

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Type falling words before they hit the bottom. Time your submissions to the beat for a score multiplier."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to navigate the virtual keyboard. Press space to type the selected key or submit your word. "
        "Press shift to toggle between upper and lower case letters."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    WIN_SCORE = 1000

    # --- Colors ---
    COLOR_BG_START = (10, 5, 25)
    COLOR_BG_END = (30, 10, 50)
    COLOR_PULSE = (60, 30, 100)
    COLOR_TEXT = (220, 220, 255)
    COLOR_WORD = (255, 255, 255)
    COLOR_TARGET_WORD = (255, 255, 100)
    COLOR_INPUT_BG = (20, 15, 40, 200)
    COLOR_INPUT_CORRECT = (100, 255, 100)
    COLOR_INPUT_INCORRECT = (255, 100, 100)
    COLOR_VK_BG = (30, 25, 60, 180)
    COLOR_VK_KEY = (80, 70, 120)
    COLOR_VK_CURSOR = (255, 255, 0)
    COLOR_VK_SHIFT = (100, 150, 255)

    WORD_LIST = [
        "agent", "reward", "action", "state", "policy", "model", "quest",
        "vector", "matrix", "tensor", "space", "field", "value", "future",
        "vision", "compute", "learn", "adapt", "train", "build", "play",
        "sync", "rhythm", "pulse", "flow", "score", "deep", "neural", "net"
    ]

    VK_LAYOUT_LOWER = [
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm<", # < is backspace
        " ^ " # ^ is submit
    ]
    VK_LAYOUT_UPPER = [
        "QWERTYUIOP",
        "ASDFGHJKL",
        "ZXCVBNM<",
        " ^ "
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_word = pygame.font.SysFont("sans", 32, bold=True)
        self.font_input = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_vk = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_feedback = pygame.font.SysFont("sans", 48, bold=True)

        self.bg_surface = self._create_gradient_background()

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This is a helper for dev, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.multiplier = 1.0
        self.word_speed = 1.0
        
        self.falling_words = []
        self.particles = []
        
        self.input_buffer = ""
        self.last_submission_status = "none" # "none", "correct", "incorrect"
        self.feedback_timer = 0

        self.vk_cursor = [0, 0]
        self.shift_active = False
        self.last_action = [0, 0, 0]

        self.rhythm_pulse = 0.0
        self.beat_frequency = 0.15 # Controls speed of the pulse
        self.submission_reward = 0

        self._spawn_word()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward_this_step = 0

        self._handle_input(action)
        self._update_game_state()

        self.steps += 1
        
        # Base reward for survival
        reward_this_step += 0.001

        # Check submission
        if self.submission_reward != 0:
            reward_this_step += self.submission_reward
        self.submission_reward = 0 # Reset after consumption

        terminated = self._check_termination()
        truncated = False # No truncation condition other than time limit, which is termination
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward_this_step += 100 # Win bonus
            else:
                reward_this_step -= 10 # Time out penalty

        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        
        # --- Movement: Navigate Virtual Keyboard ---
        if movement != 0 and movement != self.last_action[0]: # Act on change
            if movement == 1: self.vk_cursor[1] = max(0, self.vk_cursor[1] - 1)
            elif movement == 2: self.vk_cursor[1] = min(len(self.VK_LAYOUT_LOWER) - 1, self.vk_cursor[1] + 1)
            elif movement == 3: self.vk_cursor[0] = max(0, self.vk_cursor[0] - 1)
            elif movement == 4: self.vk_cursor[0] = self.vk_cursor[0] + 1
            # SFX: UI_NAVIGATE
        
        # Clamp cursor X based on current row length
        current_layout = self.VK_LAYOUT_UPPER if self.shift_active else self.VK_LAYOUT_LOWER
        self.vk_cursor[1] = min(self.vk_cursor[1], len(current_layout) - 1)
        self.vk_cursor[0] = min(self.vk_cursor[0], len(current_layout[self.vk_cursor[1]]) - 1)


        # --- Shift: Toggle Keyboard Case ---
        if shift_action == 1 and self.last_action[2] == 0: # On press
            self.shift_active = not self.shift_active
            # SFX: UI_TOGGLE

        # --- Space: Press Virtual Key ---
        self.submission_reward = 0
        if space_action == 1 and self.last_action[1] == 0: # On press
            current_layout = self.VK_LAYOUT_UPPER if self.shift_active else self.VK_LAYOUT_LOWER
            key = current_layout[self.vk_cursor[1]][self.vk_cursor[0]]
            
            if key == '<': # Backspace
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                    # SFX: KEY_PRESS_DELETE
            elif key == '^': # Submit
                self._handle_submission()
            elif key != ' ': # Type character
                self.input_buffer += key
                # SFX: KEY_PRESS_TYPE
                
                # Reward for typing a correct character in sequence
                if self.falling_words and self.input_buffer == self.falling_words[0]['text'][:len(self.input_buffer)]:
                    self.submission_reward += 0.1

        self.last_action = action

    def _handle_submission(self):
        if not self.falling_words:
            # SFX: SUBMIT_FAIL
            return

        target_word = self.falling_words[0]
        is_correct = self.input_buffer == target_word['text']
        
        if is_correct:
            # SFX: SUBMIT_CORRECT
            self.last_submission_status = "correct"
            
            # Check for rhythm sync
            is_synced = self.rhythm_pulse > 0.9
            
            # Calculate score
            base_points = len(target_word['text'])
            self.score += int(base_points * self.multiplier)
            
            # Update multiplier
            if is_synced:
                self.multiplier = min(5.0, self.multiplier + 0.5)
                # SFX: SYNC_SUCCESS
                self._create_particles(target_word['pos'], self.COLOR_INPUT_CORRECT, 30, is_synced=True)
            else:
                self.multiplier = max(1.0, self.multiplier - 0.25)
                self._create_particles(target_word['pos'], self.COLOR_INPUT_CORRECT, 15)

            # Reward
            sync_bonus = 1.0 if is_synced else 0.0
            self.submission_reward += 1.0 + sync_bonus * 0.5
            
            self.falling_words.pop(0)
            self._spawn_word()
        else:
            # SFX: SUBMIT_INCORRECT
            self.last_submission_status = "incorrect"
            self.multiplier = 1.0 # Reset multiplier on mistake
            self.score = max(0, self.score - 5)
            self.submission_reward -= 1.0
            self._create_particles((self.WIDTH / 2, self.HEIGHT - 80), self.COLOR_INPUT_INCORRECT, 20)

        self.input_buffer = ""
        self.feedback_timer = self.FPS // 2 # Show feedback for 0.5 seconds

    def _update_game_state(self):
        # Update rhythm pulse
        self.rhythm_pulse = (math.sin(self.steps * self.beat_frequency) + 1) / 2

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.word_speed = min(3.0, self.word_speed + 0.1)

        # Move words
        for word in self.falling_words[:]:
            word['pos'][1] += self.word_speed
            if word['pos'][1] > self.HEIGHT - 120:
                self.falling_words.remove(word)
                self.multiplier = 1.0 # Penalty for missed word
                # SFX: WORD_MISS
        
        # Spawn new word if screen is empty
        if not self.falling_words:
            self._spawn_word()

        # Update feedback timer
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.last_submission_status = "none"

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_word(self):
        word_text = self.np_random.choice(self.WORD_LIST)
        word_surface = self.font_word.render(word_text, True, self.COLOR_WORD)
        x_pos = self.np_random.integers(50, self.WIDTH - 50 - word_surface.get_width())
        self.falling_words.append({
            'text': word_text,
            'pos': [float(x_pos), 0.0],
            'surface': word_surface
        })

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_rhythm_pulse()
        self._render_falling_words()
        self._render_particles()
        self._render_input_field()
        self._render_virtual_keyboard()
        self._render_feedback_text()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Multiplier
        mult_color = (255, 255, 100) if self.multiplier > 1.0 else self.COLOR_TEXT
        mult_text = self.font_ui.render(f"MULT: {self.multiplier:.2f}x", True, mult_color)
        self.screen.blit(mult_text, (self.WIDTH - mult_text.get_width() - 10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

    def _render_rhythm_pulse(self):
        alpha = int(100 * self.rhythm_pulse**4)
        pulse_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pulse_surface.fill((*self.COLOR_PULSE, alpha))
        self.screen.blit(pulse_surface, (0, 0))

    def _render_falling_words(self):
        for i, word in enumerate(self.falling_words):
            color = self.COLOR_TARGET_WORD if i == 0 else self.COLOR_WORD
            text_surface = self.font_word.render(word['text'], True, color)
            pos = (int(word['pos'][0]), int(word['pos'][1]))
            self.screen.blit(text_surface, pos)

    def _render_input_field(self):
        field_rect = pygame.Rect(50, self.HEIGHT - 90, self.WIDTH - 100, 40)
        
        # Feedback flash
        if self.feedback_timer > 0:
            color = self.COLOR_INPUT_CORRECT if self.last_submission_status == "correct" else self.COLOR_INPUT_INCORRECT
            pygame.draw.rect(self.screen, color, field_rect, border_radius=5)
        
        pygame.draw.rect(self.screen, self.COLOR_INPUT_BG, field_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, field_rect, 2, border_radius=5)

        input_text = self.input_buffer
        # Blinking cursor
        if (self.steps // (self.FPS // 2)) % 2 == 0:
            input_text += "_"

        text_surface = self.font_input.render(input_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=field_rect.center)
        self.screen.blit(text_surface, text_rect)

    def _render_virtual_keyboard(self):
        vk_h = 90
        vk_y = self.HEIGHT - vk_h
        
        # Background
        bg_rect = pygame.Rect(0, vk_y, self.WIDTH, vk_h)
        pygame.draw.rect(self.screen, self.COLOR_VK_BG, bg_rect)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, vk_y), (self.WIDTH, vk_y), 1)

        # Keys
        current_layout = self.VK_LAYOUT_UPPER if self.shift_active else self.VK_LAYOUT_LOWER
        key_h = 22
        
        for r, row in enumerate(current_layout):
            key_w = (self.WIDTH - 40) / len(row) if len(row) > 1 else self.WIDTH / 2
            for c, key in enumerate(row):
                if key == ' ': continue
                
                is_shift = r == 3 and c == 0
                is_submit = r == 3 and c == 1
                
                key_x = 20 + c * key_w
                key_y = vk_y + 2 + r * key_h
                
                if is_shift:
                    w = key_w * 3
                    text = "SHIFT"
                elif is_submit:
                    key_x = 20 + 4 * key_w
                    w = self.WIDTH - 40 - key_x + 20
                    text = "SUBMIT"
                else:
                    w = key_w
                    text = "DEL" if key == '<' else key
                
                key_rect = pygame.Rect(key_x, key_y, w - 2, key_h - 2)
                
                key_color = self.COLOR_VK_SHIFT if (self.shift_active and is_shift) else self.COLOR_VK_KEY
                pygame.draw.rect(self.screen, key_color, key_rect, border_radius=3)
                
                text_surf = self.font_vk.render(text, True, self.COLOR_TEXT)
                self.screen.blit(text_surf, text_surf.get_rect(center=key_rect.center))

        # Cursor
        cursor_r, cursor_c = self.vk_cursor[1], self.vk_cursor[0]
        row_str = current_layout[cursor_r]
        key_w = (self.WIDTH - 40) / len(row_str) if len(row_str) > 1 else self.WIDTH / 2
        
        is_shift = cursor_r == 3 and cursor_c == 0
        is_submit = cursor_r == 3 and cursor_c == 1
        
        if is_shift:
            w = key_w * 3
            x = 20
        elif is_submit:
            w = self.WIDTH - 40 - (20 + 4 * key_w)
            x = 20 + 4 * key_w
        else:
            w = key_w
            x = 20 + cursor_c * key_w
            
        y = vk_y + 2 + cursor_r * key_h
        cursor_rect = pygame.Rect(x, y, w - 2, key_h - 2)
        pygame.draw.rect(self.screen, self.COLOR_VK_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_feedback_text(self):
        if self.feedback_timer > 0 and self.last_submission_status == "correct":
            is_synced = self.multiplier > getattr(self, 'prev_multiplier', self.multiplier - 1)
            if is_synced:
                text = "SYNC!"
                color = self.COLOR_TARGET_WORD
                pos = (self.WIDTH // 2, self.HEIGHT // 2 - 50)
                alpha = int(255 * (self.feedback_timer / (self.FPS / 2)))
                
                feedback_surf = self.font_feedback.render(text, True, color)
                feedback_surf.set_alpha(alpha)
                self.screen.blit(feedback_surf, feedback_surf.get_rect(center=pos))

        self.prev_multiplier = self.multiplier

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] * 0.2)
            if radius > 0:
                color = p['color'] if not p['is_synced'] else (
                    self.np_random.integers(200, 256), self.np_random.integers(200, 256), self.np_random.integers(100, 201)
                )
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _create_particles(self, pos, color, count, is_synced=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5) if is_synced else self.np_random.uniform(0.5, 2.5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 41),
                'color': color,
                'is_synced': is_synced
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "multiplier": self.multiplier,
            "word_speed": self.word_speed,
            "target_word": self.falling_words[0]['text'] if self.falling_words else ""
        }
        
    def _create_gradient_background(self):
        surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio,
                self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio,
                self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio,
            )
            pygame.draw.line(surf, color, (0, y), (self.WIDTH, y))
        return surf

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block is for human play and debugging.
    # It will not be executed by the autograder, but is useful for development.
    # To run, you'll need to `pip install pygame`.
    # It also requires a display, so it will not run in a headless environment.
    # To run headlessly, you can comment out this block or ensure SDL_VIDEODRIVER is set to "dummy".
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Wordfall Rhapsody - Human Player")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Input to Action Mapping ---
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                elif event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for reset key
            
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()