import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:47:26.149352
# Source Brief: brief_01281.md
# Brief Index: 1281
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced typing challenge. Select letters on the virtual keyboard to spell the target word and submit it before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor on the keyboard. Press 'space' to type the selected character and 'shift' to submit the word."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3600 # 2 minutes at 30 FPS
    WIN_SCORE = 200
    FPS = 30

    # Colors
    COLOR_BG_TOP = (15, 20, 45)
    COLOR_BG_BOTTOM = (30, 40, 70)
    COLOR_TEXT = (220, 220, 255)
    COLOR_CORRECT = (100, 255, 120)
    COLOR_INCORRECT = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_GLOW = (255, 255, 0, 50)
    COLOR_KEYBOARD_BG = (50, 60, 90)
    COLOR_KEYBOARD_KEY = (80, 90, 120)
    COLOR_TIMER_BAR = (60, 180, 255)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_DANGER = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 52)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Word List ---
        self.word_list = [
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'ANY', 'CAN',
            'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS',
            'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY',
            'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'CODE', 'GAME',
            'PLAY', 'WORK', 'TIME', 'FAST', 'SLOW', 'GOOD', 'BEST', 'TEST', 'DEEP',
            'MIND', 'GYM', 'PYTHON', 'AGENT', 'ACTION', 'REWARD', 'STATE', 'POLICY',
            'LEARN', 'TRAIN', 'MODEL', 'VISUAL', 'EFFECT', 'QUICK', 'BROWN', 'FOX',
            'JUMPS', 'OVER', 'LAZY', 'DOG', 'ARCADE', 'TYPING', 'CHALLENGE', 'EXPERT',
            'FLUID', 'MOTION', 'ACCURATE', 'FEEDBACK', 'MINIMALIST', 'GRADIENT'
        ]

        # --- Keyboard Layout ---
        self.keyboard_layout = [
            "QWERTYUIOP",
            "ASDFGHJKL",
            "ZXCVBNM< "  # < is backspace, ' ' is space
        ]
        self.key_positions = self._generate_key_positions()
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.current_word = ""
        self.input_string = ""
        self.cursor_pos = (0, 0)
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_messages = []

    
    def _generate_key_positions(self):
        positions = {}
        key_size = 38
        key_margin = 4
        total_key_width = key_size + key_margin
        
        start_y = self.SCREEN_HEIGHT - 3 * total_key_width + key_margin / 2 - 10
        
        for r, row_str in enumerate(self.keyboard_layout):
            row_width = len(row_str) * total_key_width
            start_x = (self.SCREEN_WIDTH - row_width) / 2
            for c, char in enumerate(row_str):
                x = start_x + c * total_key_width
                y = start_y + r * total_key_width
                positions[(r, c)] = pygame.Rect(int(x), int(y), key_size, key_size)
        return positions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 120 * self.FPS  # 2 minutes
        self.input_string = ""
        self.cursor_pos = (0, 0)
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_messages = []
        
        self._generate_new_word()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for taking time

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        
        key_typed = space_held and not self.last_space_held
        word_submitted = shift_held and not self.last_shift_held

        if key_typed:
            char_to_type = self.keyboard_layout[self.cursor_pos[0]][self.cursor_pos[1]]
            # # Sound: Key press
            if char_to_type == '<': # Backspace
                if len(self.input_string) > 0:
                    self.input_string = self.input_string[:-1]
            else:
                if len(self.input_string) < 20: # Limit input length
                    self.input_string += char_to_type
            
            # Per-character reward
            if len(self.input_string) <= len(self.current_word) and self.current_word.startswith(self.input_string):
                reward += 0.1
            else:
                reward -= 0.2

        if word_submitted:
            reward += self._check_submission()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game State ---
        self.timer = max(0, self.timer - 1)
        self._update_particles()
        self._update_feedback_messages()

        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            self.feedback_messages.append({"text": "YOU WIN!", "pos": (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50), "color": self.COLOR_CORRECT, "life": self.FPS * 3, "font": self.font_large})
            terminated = True
        elif self.timer <= 0:
            reward -= 50
            self.feedback_messages.append({"text": "TIME UP!", "pos": (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50), "color": self.COLOR_INCORRECT, "life": self.FPS * 3, "font": self.font_large})
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        r, c = self.cursor_pos
        if movement == 1:  # Up
            r = max(0, r - 1)
        elif movement == 2:  # Down
            r = min(len(self.keyboard_layout) - 1, r + 1)
        elif movement == 3:  # Left
            c = max(0, c - 1)
        elif movement == 4:  # Right
            c = min(len(self.keyboard_layout[r]) - 1, c + 1)
        
        # Adjust for jagged rows
        c = min(c, len(self.keyboard_layout[r]) - 1)
        self.cursor_pos = (r, c)
        
    def _check_submission(self):
        reward = 0
        # Use strip() to handle the space key being used for submission
        if self.input_string.strip() == self.current_word:
            # # Sound: Correct word
            reward = 10
            self.score += len(self.current_word)
            self._create_feedback_message("CORRECT!", self.COLOR_CORRECT)
            self._create_particles(self.COLOR_CORRECT)
        else:
            # # Sound: Incorrect word
            reward = -5
            self.score = max(0, self.score - len(self.current_word) // 2)
            self._create_feedback_message("INCORRECT", self.COLOR_INCORRECT)
            self._create_particles(self.COLOR_INCORRECT)
        
        self.input_string = ""
        self._generate_new_word()
        return reward

    def _generate_new_word(self):
        min_len = 3 + (self.score // 50)
        max_len = min_len + 2
        
        possible_words = [w for w in self.word_list if min_len <= len(w) <= max_len]
        if not possible_words:
             possible_words = [w for w in self.word_list if len(w) >= min_len]
        if not possible_words: # Fallback
            possible_words = self.word_list

        new_word = random.choice(possible_words)
        while new_word == self.current_word and len(possible_words) > 1:
             new_word = random.choice(possible_words)
        self.current_word = new_word

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer / self.FPS}

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), p["size"], p["size"]))

        # Feedback Messages
        for msg in self.feedback_messages:
            alpha = min(255, int(255 * (msg["life"] / (self.FPS * 0.5))))
            text_surf = msg["font"].render(msg["text"], True, msg["color"])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=msg["pos"])
            self.screen.blit(text_surf, text_rect)

        # Target Word
        target_surf = self.font_large.render(self.current_word, True, self.COLOR_TEXT)
        target_rect = target_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 80))
        self.screen.blit(target_surf, target_rect)
        
        # Input String
        self._render_input_string()

        # Keyboard
        self._render_keyboard()

    def _render_input_string(self):
        x_offset_total = 0
        base_y = 140
        
        # Pre-calculate total width to center the string
        for char in self.input_string:
            char_surf = self.font_medium.render(char, True, self.COLOR_TEXT)
            x_offset_total += char_surf.get_width()
            
        start_x = self.SCREEN_WIDTH // 2 - x_offset_total / 2
        current_x = start_x
        
        for i, char in enumerate(self.input_string):
            is_correct = i < len(self.current_word) and self.current_word[i] == char
            color = self.COLOR_CORRECT if is_correct else self.COLOR_INCORRECT
            char_surf = self.font_medium.render(char, True, color)
            char_rect = char_surf.get_rect(midleft=(current_x, base_y))
            self.screen.blit(char_surf, char_rect)
            current_x += char_surf.get_width()

        # Blinking cursor
        if (self.steps // (self.FPS // 2)) % 2 == 0:
            cursor_rect = pygame.Rect(int(current_x + 2), int(base_y - 15), 3, 30)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect)

    def _render_keyboard(self):
        for (r, c), rect in self.key_positions.items():
            pygame.draw.rect(self.screen, self.COLOR_KEYBOARD_BG, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_KEYBOARD_KEY, rect, width=2, border_radius=5)
            
            char = self.keyboard_layout[r][c]
            if char == '<': char_text = "DEL"
            elif char == ' ': char_text = "SPACE"
            else: char_text = char

            char_surf = self.font_medium.render(char_text, True, self.COLOR_TEXT)
            char_rect = char_surf.get_rect(center=rect.center)
            self.screen.blit(char_surf, char_rect)

        # Cursor
        cursor_rect = self.key_positions[self.cursor_pos]
        
        # Glow effect
        glow_rect = cursor_rect.inflate(12, 12)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.gfxdraw.box(s, (0, 0, glow_rect.width, glow_rect.height), self.COLOR_CURSOR_GLOW)
        self.screen.blit(s, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=5)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 15))

        # Timer Bar
        timer_ratio = self.timer / (120 * self.FPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        timer_color = self.COLOR_TIMER_BAR
        if timer_ratio < 0.5: timer_color = self.COLOR_TIMER_WARN
        if timer_ratio < 0.2: timer_color = self.COLOR_TIMER_DANGER
        
        pygame.draw.rect(self.screen, self.COLOR_KEYBOARD_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height), border_radius=5)
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # Gravity
            p["life"] -= 1

    def _create_particles(self, color):
        center_x, center_y = self.SCREEN_WIDTH // 2, 110
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(20, 40),
                "color": color,
                "size": random.randint(3, 6)
            })

    def _update_feedback_messages(self):
        self.feedback_messages = [m for m in self.feedback_messages if m["life"] > 0]
        for msg in self.feedback_messages:
            msg["life"] -= 1

    def _create_feedback_message(self, text, color):
        self.feedback_messages.append({
            "text": text,
            "pos": (self.SCREEN_WIDTH // 2, 180),
            "color": color,
            "life": self.FPS, # 1 second
            "font": self.font_medium
        })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # The original code had a validation function that is not part of the standard API.
    # It has been removed to avoid confusion and ensure compatibility.
    # The main execution block is for manual testing and demonstration.
    
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrow keys: Move cursor
    # Space: Type selected character
    # Left Shift: Submit word
    # R: Reset environment
    # Q: Quit
    
    # This mapping is for human play, not the agent's action space.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Use a separate screen for rendering if playing manually
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Typing Challenge Gym Environment")
    
    clock = pygame.time.Clock()
    total_reward = 0
    
    # Store button states for MultiDiscrete action
    movement_action = 0
    space_action = 0
    shift_action = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_SPACE:
                    space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_action = 1
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
                if event.key == pygame.K_q:
                    running = False

            if event.type == pygame.KEYUP:
                if event.key in key_to_action and movement_action == key_to_action[event.key]:
                    movement_action = 0
                if event.key == pygame.K_SPACE:
                    space_action = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_action = 0

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()